-- kms database (owner: kms). Only the KMS service can connect (pg_hba + REVOKE CONNECT FROM PUBLIC).
CREATE TABLE subject_keys (
  subject_id   uuid PRIMARY KEY,
  wrapped_key  bytea,                     -- nonce(12) || AES-256-GCM(master key, data key, aad=subject_id); NULL = shredded
  created_at   timestamptz NOT NULL DEFAULT now(),
  destroyed_at timestamptz,
  CHECK ((wrapped_key IS NULL) = (destroyed_at IS NOT NULL))
);
