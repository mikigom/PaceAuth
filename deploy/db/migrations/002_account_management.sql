-- Run as faceauth_api before deploying API account-management routes. Safe to re-run.
CREATE TABLE IF NOT EXISTS account_controls (
  account_id       text PRIMARY KEY,
  enabled          boolean NOT NULL DEFAULT true,
  enrollment_epoch bigint NOT NULL DEFAULT 0,
  updated_at       timestamptz NOT NULL DEFAULT now()
);
