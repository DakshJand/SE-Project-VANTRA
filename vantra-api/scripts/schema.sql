-- VANTRA schema. PostGIS enabled database required.

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lng         DOUBLE PRECISION NOT NULL,
    geom        geometry(Point, 4326),
    road        TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id     TEXT PRIMARY KEY,          -- "CAM01->CAM02"
    src_cam     TEXT NOT NULL REFERENCES cameras,
    dst_cam     TEXT NOT NULL REFERENCES cameras,
    distance_m  DOUBLE PRECISION NOT NULL,
    speed_limit_kmph INTEGER NOT NULL,
    stop_eligible BOOLEAN NOT NULL DEFAULT FALSE,  -- parking/market/fuel: wider grace window
    min_time_s  DOUBLE PRECISION NOT NULL,          -- hard floor: distance / (limit * 1.6)
    -- learned from historical logs (fitted during simulation):
    tt_p5       DOUBLE PRECISION,
    tt_p50      DOUBLE PRECISION,
    tt_p95      DOUBLE PRECISION,
    tt_mean     DOUBLE PRECISION,
    tt_std      DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS edges_src_idx ON edges(src_cam);
CREATE INDEX IF NOT EXISTS edges_dst_idx ON edges(dst_cam);

-- Each row = one fitted (edge, hop-reach) travel-time distribution.
CREATE TABLE IF NOT EXISTS edge_time_stats (
    edge_id     TEXT NOT NULL,
    hop_reach   INTEGER NOT NULL,          -- 1 = direct edge, 2 = skip-one-hop window
    p5          DOUBLE PRECISION NOT NULL,
    p50         DOUBLE PRECISION NOT NULL,
    p95         DOUBLE PRECISION NOT NULL,
    mean        DOUBLE PRECISION NOT NULL,
    std         DOUBLE PRECISION NOT NULL,
    n_samples   INTEGER NOT NULL,
    PRIMARY KEY (edge_id, hop_reach)
);

-- Raw ANPR detections as emitted by the camera pipeline.
CREATE TABLE IF NOT EXISTS detections (
    det_id          BIGSERIAL PRIMARY KEY,
    camera_id       TEXT NOT NULL REFERENCES cameras,
    ts              TIMESTAMPTZ NOT NULL,
    plate_raw       TEXT,                  -- OCR output; NULL/empty if unreadable
    ocr_conf        DOUBLE PRECISION,      -- OCR confidence [0,1]
    plate_correct   TEXT,                  -- ground-truth plate (simulation only; NULL in prod)
    vehicle_type    TEXT,                  -- car / suv / truck / bus / auto / bike
    vehicle_color   TEXT,                  -- normalized color name
    crop_path       TEXT,                  -- cropped vehicle image
    plate_crop_path TEXT,                  -- cropped plate image
    embed           DOUBLE PRECISION[],    -- appearance embedding (Re-ID)
    session_id      TEXT,                  -- simulation ground-truth trip id
    gt_cam          TEXT                   -- ground truth: camera it actually passed
);
CREATE INDEX IF NOT EXISTS det_plate_ts_idx ON detections(plate_raw, ts);
CREATE INDEX IF NOT EXISTS det_cam_ts_idx ON detections(camera_id, ts);
CREATE INDEX IF NOT EXISTS det_ts_idx ON detections(ts);
CREATE INDEX IF NOT EXISTS det_session_idx ON detections(session_id);

-- Trajectories: a chain of detections for one (reconstructed) vehicle.
CREATE TABLE IF NOT EXISTS trajectories (
    traj_id     BIGSERIAL PRIMARY KEY,
    plate       TEXT NOT NULL,             -- canonical plate of the trajectory
    t_start     TIMESTAMPTZ NOT NULL,
    t_end       TIMESTAMPTZ NOT NULL,
    confidence  DOUBLE PRECISION NOT NULL, -- rollup confidence (not a plain average)
    n_hops      INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS traj_plate_idx ON trajectories(plate);

-- One hop = link between two consecutive detections in a trajectory.
CREATE TABLE IF NOT EXISTS trajectory_hops (
    hop_id          BIGSERIAL PRIMARY KEY,
    traj_id         BIGINT NOT NULL REFERENCES trajectories ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    from_det_id     BIGINT NOT NULL REFERENCES detections,
    to_det_id       BIGINT NOT NULL REFERENCES detections,
    tier            TEXT NOT NULL,         -- exact|fuzzy_unique|fuzzy_disambiguated|appearance_only
    confidence      DOUBLE PRECISION NOT NULL,
    time_score      DOUBLE PRECISION,      -- travel-time percentile-based score
    appearance_score DOUBLE PRECISION,     -- cosine similarity (if available)
    edit_distance   INTEGER,               -- plate edit distance (if plate matched)
    ocr_conf_min    DOUBLE PRECISION,      -- weaker OCR conf of the pair
    edge_id         TEXT,
    evidence        JSONB NOT NULL         -- full machine-readable reasoning
);
CREATE INDEX IF NOT EXISTS hop_traj_idx ON trajectory_hops(traj_id);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id    BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL,             -- blacklist|hard_anomaly|soft_anomaly|convoy
    severity    TEXT NOT NULL,             -- high|medium|low
    plate       TEXT,
    title       TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    det_ids     BIGINT[],
    evidence    JSONB NOT NULL,
    reviewed    BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS alerts_kind_ts_idx ON alerts(kind, ts);

CREATE TABLE IF NOT EXISTS blacklist (
    plate       TEXT PRIMARY KEY,
    reason      TEXT NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS demo_scenarios (
    key         TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    plate       TEXT,
    payload     JSONB NOT NULL             -- pointers used by demo runner / dashboard
);
