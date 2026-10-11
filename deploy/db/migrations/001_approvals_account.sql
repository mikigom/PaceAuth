-- 001: the approval ledger also records non-biometric (fallback) approvals of accounts without a face enrollment.
-- faceauth.sql already contains this; run on databases created before 2026-10-11 (as faceauth_api).
ALTER TABLE approvals ALTER COLUMN subject_id DROP NOT NULL;
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS account_id text;
UPDATE approvals a SET account_id = s.account_id FROM subjects s WHERE a.account_id IS NULL AND s.id = a.subject_id;
ALTER TABLE approvals ALTER COLUMN account_id SET NOT NULL;
