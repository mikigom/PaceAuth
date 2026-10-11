-- faceauth database (owner: faceauth_api). Read-only role faceauth_ro is used by engine/search.
-- Contract shared by api (writer), search (template reader) and engine (model_versions reader).

CREATE TABLE devices (
  id               text PRIMARY KEY,
  public_key       bytea NOT NULL,              -- Ed25519 raw 32 bytes; verifies capture packets
  cert_fingerprint text NOT NULL UNIQUE,        -- sha256(DER of mTLS client cert), lowercase hex
  calib_version    text NOT NULL,               -- packets must carry this exact calibration version
  status           text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE subjects (                         -- id is the pseudonymous key; only it is stored with templates
  id               uuid PRIMARY KEY,
  account_id       text NOT NULL,
  status           text NOT NULL CHECK (status IN ('active', 'pending_review', 'revoked')),
  consent_version  text NOT NULL,
  consent_at       timestamptz NOT NULL,
  template_version int NOT NULL DEFAULT 1,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  revoked_at       timestamptz
);
CREATE UNIQUE INDEX subjects_live_account ON subjects (account_id) WHERE status <> 'revoked';

CREATE TABLE templates (
  id               bigserial PRIMARY KEY,
  subject_id       uuid NOT NULL REFERENCES subjects (id),
  template_version int NOT NULL,
  slot             smallint NOT NULL,           -- 0..3
  model_version    text NOT NULL,               -- "<model name>@<version>"; never compared across versions
  align_version    text NOT NULL,
  calib_version    text NOT NULL,
  ciphertext       bytea,                       -- KMS AES-256-GCM(float32[d] little-endian); NULL once deleted
  active           boolean NOT NULL DEFAULT true,
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX templates_active ON templates (model_version, id) WHERE active;
CREATE INDEX templates_subject ON templates (subject_id);

-- Outbox consumed by search replicas. Writers MUST run `SELECT pg_advisory_xact_lock(7001)` before
-- inserting here so event ids become visible in commit order (no gaps skipped by pollers).
CREATE TABLE template_events (
  id               bigserial PRIMARY KEY,
  op               text NOT NULL CHECK (op IN ('add', 'remove')),
  template_id      bigint NOT NULL,
  subject_id       uuid NOT NULL,
  model_version    text NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
  id               uuid PRIMARY KEY,
  purpose          text NOT NULL CHECK (purpose IN ('enroll', 'auth')),
  device_id        text NOT NULL REFERENCES devices (id),
  action_id        text NOT NULL,
  challenge        text NOT NULL,               -- b64url(32 random bytes)
  created_at       timestamptz NOT NULL DEFAULT now(),
  expires_at       timestamptz NOT NULL,
  consumed_at      timestamptz,                 -- single use; set in the same tx as the decision
  decision         text,                        -- approved|rejected|step_up|enrolled|pending_review
  reason           text                         -- internal detail (never returned verbatim)
);
CREATE INDEX sessions_device_time ON sessions (device_id, created_at);

CREATE TABLE approvals (                        -- every approval, biometric or not (fallback: no subject)
  id               uuid PRIMARY KEY,
  session_id       uuid NOT NULL UNIQUE REFERENCES sessions (id),
  subject_id       uuid REFERENCES subjects (id),
  account_id       text NOT NULL,
  device_id        text NOT NULL,
  action_id        text NOT NULL,
  amr              text[] NOT NULL,             -- {face} | {face,otp} | {pin} ...; face alone is never "mfa"
  issued_at        timestamptz NOT NULL DEFAULT now(),
  expires_at       timestamptz NOT NULL
);

CREATE TABLE stepups (
  id               uuid PRIMARY KEY,
  session_id       uuid NOT NULL UNIQUE REFERENCES sessions (id),
  subject_id       uuid NOT NULL REFERENCES subjects (id),
  device_id        text NOT NULL,
  action_id        text NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  expires_at       timestamptz NOT NULL,
  completed_at     timestamptz
);

CREATE TABLE enrollment_reviews (               -- duplicate candidates; never merged automatically
  id                   uuid PRIMARY KEY,
  subject_id           uuid NOT NULL REFERENCES subjects (id),
  candidate_subject_id uuid NOT NULL REFERENCES subjects (id),
  status               text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'approved', 'rejected')),
  created_at           timestamptz NOT NULL DEFAULT now(),
  decided_at           timestamptz,
  decided_by           text
);

CREATE TABLE model_versions (
  name             text NOT NULL,
  version          text NOT NULL,
  sha256           text NOT NULL,               -- of the signed registry entry's primary file
  active           boolean NOT NULL DEFAULT false,
  activated_at     timestamptz,
  PRIMARY KEY (name, version)
);
CREATE UNIQUE INDEX model_versions_one_active ON model_versions (name) WHERE active;

-- Append-only, hash-chained audit log. Insert it as the LAST statement of a transaction:
-- the chain lock is held until commit.
CREATE SEQUENCE audit_log_seq;
CREATE TABLE audit_log (
  id               bigint PRIMARY KEY,
  ts               timestamptz NOT NULL DEFAULT clock_timestamp(),
  actor            text NOT NULL,
  event            text NOT NULL,
  subject_id       uuid,
  session_id       uuid,
  device_id        text,
  detail           jsonb NOT NULL DEFAULT '{}',
  prev_hash        bytea NOT NULL,
  hash             bytea NOT NULL
);

CREATE FUNCTION audit_row_hash(prev bytea, r audit_log) RETURNS bytea LANGUAGE sql IMMUTABLE AS $$
  SELECT sha256(prev || convert_to(concat_ws('|', r.id, extract(epoch FROM r.ts)::numeric, r.actor, r.event,
    coalesce(r.subject_id::text, ''), coalesce(r.session_id::text, ''), coalesce(r.device_id, ''), r.detail::text), 'UTF8'))
$$;

CREATE FUNCTION audit_chain() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE prev bytea;
BEGIN
  PERFORM pg_advisory_xact_lock(7002);
  NEW.id := nextval('audit_log_seq');            -- assigned under the lock => id order == chain order
  SELECT hash INTO prev FROM audit_log ORDER BY id DESC LIMIT 1;
  NEW.prev_hash := coalesce(prev, '\x'::bytea);
  NEW.hash := audit_row_hash(NEW.prev_hash, NEW);
  RETURN NEW;
END $$;
CREATE TRIGGER audit_chain BEFORE INSERT ON audit_log FOR EACH ROW EXECUTE FUNCTION audit_chain();

CREATE FUNCTION audit_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'audit_log is append-only'; END $$;
CREATE TRIGGER audit_no_change BEFORE UPDATE OR DELETE ON audit_log FOR EACH ROW EXECUTE FUNCTION audit_immutable();
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log EXECUTE FUNCTION audit_immutable();

-- Returns NULL if the whole chain verifies, else the first bad id.
CREATE FUNCTION audit_verify() RETURNS bigint LANGUAGE plpgsql STABLE AS $$
DECLARE r audit_log; prev bytea := '\x';
BEGIN
  FOR r IN SELECT * FROM audit_log ORDER BY id LOOP
    IF r.prev_hash <> prev OR r.hash <> audit_row_hash(prev, r) THEN RETURN r.id; END IF;
    prev := r.hash;
  END LOOP;
  RETURN NULL;
END $$;
