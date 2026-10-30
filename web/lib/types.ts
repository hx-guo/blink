// 与 `blink export` 的输出一一对应（crates/workflows/blink/src/commands/export.rs）

export type InstrumentSummary = {
  instrument: string;
  slug: string;
  public: boolean;
  demo: boolean;
  n_events: number;
  n_final: number;
  n_named: number;
  electron_beams: number;
  first_day: string;
  last_day: string;
  searched_seconds: number;
  generated_at: string;
  blink_git: string;
};

export type Index = {
  generated_at: string;
  public_only: boolean;
  n_names: number;
  instruments: InstrumentSummary[];
};

export type LatestEvent = {
  name: string | null;
  cand_id: string;
  instrument: string;
  slug: string;
  status: "final" | "provisional";
  peak: string;
  longitude: number;
  latitude: number;
  false_positive_per_year: number;
  count: number;
  duration_us: number;
};

export type Detection = {
  instrument: string;
  slug: string;
  cand_id: string | null;
  start: string;
  peak: string;
  longitude: number | null;
  latitude: number | null;
};

export type NameEntry = {
  name: string;
  peak_utc: string;
  assigned_at: string;
  status: "active" | "retracted" | "merged";
  status_reason: string;
  merged_into: string;
  detections: Detection[];
};

/** 列式目录：每列一个等长数组 */
export type Catalog = {
  instrument: string;
  slug: string;
  n: number;
  columns: Record<string, (string | number | null)[]>;
};

export type LightCurve = {
  peak: string;
  columns: ["t_us", "det", "channel", "acd"];
  events: [number, number, number, number][];
  context_bin_s: number;
  context_start_s: number;
  context: number[];
};

export type EventRecord = {
  instrument: string;
  slug: string;
  name: string | null;
  cand_id: string;
  status: "final" | "provisional";
  pending: string | null;
  start: string;
  stop: string;
  peak: string;
  duration_us: number;
  count: number;
  false_positive_per_year: number;
  longitude: number;
  latitude: number;
  altitude_km: number;
  /** 星下点的视太阳时 HH:MM:SS */
  solar_time?: string;
  neighbors_10min: number | null;
  grb_z: number | null;
  grb_max_box_share: number | null;
  electron_beam: number | null;
  beam_p: number | null;
  det_share_max: number | null;
  n: number | null;
  n_acd: number | null;
  n_acd_multi: number | null;
  n_bg: number | null;
  n_acd_bg: number | null;
  light_curve: LightCurve | null;
};
