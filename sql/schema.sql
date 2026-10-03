-- Kinetic calibration service schema (PostgreSQL 16)

CREATE TABLE IF NOT EXISTS networks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    definition  JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS datasets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id  UUID NOT NULL REFERENCES networks(id),
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_datasets_network ON datasets(network_id);

-- each insert/remove of a batch produces a new immutable version
CREATE TABLE IF NOT EXISTS dataset_versions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id  UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    parent_version_id UUID REFERENCES dataset_versions(id),
    change_note TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_dataset ON dataset_versions(dataset_id);

CREATE TABLE IF NOT EXISTS batches (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id  UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    temperature DOUBLE PRECISION NOT NULL,
    initial_concentrations JSONB NOT NULL,
    samples     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_batches_version ON batches(version_id);

CREATE TABLE IF NOT EXISTS calibrations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id      UUID NOT NULL REFERENCES networks(id),
    version_id      UUID NOT NULL REFERENCES dataset_versions(id),
    parent_calibration_id UUID REFERENCES calibrations(id),
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_calibrations_version ON calibrations(version_id);
