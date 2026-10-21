mod cli;
mod commands;
mod util;

use clap::Parser;

use cli::{Cli, DumpCommands, Instrument, SatCommands, TopCommands};
use commands::compare::cmd_compare;
use commands::detect::cmd_detect;
use commands::dump::{
    cmd_dump_check_offset, cmd_dump_diag, cmd_dump_events, cmd_dump_hist, cmd_dump_packets,
    cmd_dump_ptime, cmd_dump_times,
};
use commands::extract::{cmd_extract_1b, cmd_extract_1k};
use commands::inject::cmd_inject;
use commands::reconstruct::cmd_reconstruct;
use commands::report::cmd_report;
use util::{filter_boxes, load_boxes, parse_epoch, warn_if_window_crosses_hour};

fn main() {
    let cli = Cli::parse();

    match cli.command {
        TopCommands::Sat { command } => match command {
            SatCommands::Report(args) => {
                cmd_report(&args).expect("report failed");
            }
            SatCommands::Detect(args) => {
                let epoch = args.window.epoch();
                let met = args.window.trigger_met();
                warn_if_window_crosses_hour(met, args.window.before, args.window.after, epoch);
                eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                let boxes = load_boxes(epoch);
                let filtered = filter_boxes(&boxes, &args.window.box_filter);
                cmd_detect(&filtered, Some(args.window.met_min()), Some(args.window.met_max()));
            }
            SatCommands::Reconstruct(args) => {
                let epoch = args.window.epoch();
                let met = args.window.trigger_met();
                warn_if_window_crosses_hour(met, args.window.before, args.window.after, epoch);
                eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                let boxes = load_boxes(epoch);
                let filter_box = args.window.box_filter.clone();
                cmd_reconstruct(&args, &boxes, &filter_box);
            }
            SatCommands::Inject(args) => {
                let epoch = args.window.epoch();
                let met = args.window.trigger_met();
                warn_if_window_crosses_hour(met, args.window.before, args.window.after, epoch);
                eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                let boxes = load_boxes(epoch);
                cmd_inject(&args, &boxes);
            }
            SatCommands::Extract(args) => {
                let epoch = args.window.epoch();
                let met = args.window.trigger_met();
                warn_if_window_crosses_hour(met, args.window.before, args.window.after, epoch);
                match args.source.as_str() {
                    "1b" => {
                        eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                        let boxes = load_boxes(epoch);
                        let filtered = filter_boxes(&boxes, &args.window.box_filter);
                        cmd_extract_1b(&filtered, args.window.met_min(), args.window.met_max());
                    }
                    "1k" => {
                        cmd_extract_1k(epoch, &args.window.box_filter,
                                       args.window.met_min(), args.window.met_max());
                    }
                    other => {
                        eprintln!("error: --source must be '1b' or '1k', got '{}'", other);
                        std::process::exit(2);
                    }
                }
            }
            SatCommands::Compare(args) => {
                let epoch = args.window.epoch();
                let met = args.window.trigger_met();
                warn_if_window_crosses_hour(met, args.window.before, args.window.after, epoch);
                eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                let boxes = load_boxes(epoch);
                let filter_box = args.window.box_filter.clone();
                cmd_compare(&args, &boxes, epoch, &filter_box);
            }
            SatCommands::Scan(args) => {
                let epoch = parse_epoch(&args.epoch);
                eprintln!("Loading 1B files for {}...", epoch.format("%Y-%m-%dT%H"));
                let boxes = load_boxes(epoch);
                let filtered = filter_boxes(&boxes, &args.box_filter);
                cmd_detect(&filtered, None, None);
            }
            SatCommands::Dump { sub } => match sub {
                DumpCommands::Times(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_times(&a, &filtered);
                }
                DumpCommands::Packets(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_packets(&a, &filtered, &boxes);
                }
                DumpCommands::Events(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_events(&a, &filtered);
                }
                DumpCommands::Hist(a) => {
                    let epoch = parse_epoch(&a.window.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.window.box_filter);
                    cmd_dump_hist(&a, &filtered);
                }
                DumpCommands::Diag(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_diag(&a, &filtered);
                }
                DumpCommands::Ptime(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_ptime(&a.epoch, a.pkt_min, a.pkt_max, &filtered);
                }
                DumpCommands::CheckOffset(a) => {
                    let epoch = parse_epoch(&a.epoch);
                    let boxes = load_boxes(epoch);
                    let filtered = filter_boxes(&boxes, &a.box_filter);
                    cmd_dump_check_offset(&a.epoch, a.pkt_min, a.pkt_max, &filtered);
                }
            },
        },
        TopCommands::Search {
            from,
            to,
            workers,
            worker,
            instrument,
        } => {
            let start = chrono::NaiveDate::parse_from_str(&from, "%Y-%m-%d")
                .unwrap_or_else(|e| panic!("invalid --from date '{from}': {e}"));
            let end = chrono::NaiveDate::parse_from_str(&to, "%Y-%m-%d")
                .unwrap_or_else(|e| panic!("invalid --to date '{to}': {e}"));
            assert!(workers >= 1, "--workers must be >= 1");
            assert!(
                worker < workers,
                "--worker {worker} out of range [0, {workers})"
            );
            eprintln!(
                "TGF search {start} .. {end}  (worker {worker}/{workers}, {instrument:?})",
            );
            match instrument {
                Instrument::HxmtHe => blink_search::search_range::<blink_hxmt_he::types::HxmtHe>(
                    start, end, workers, worker,
                ),
                Instrument::SvomGrm => blink_search::search_range::<blink_svom_grm::types::SvomGrm>(
                    start, end, workers, worker,
                ),
                Instrument::FermiGbm => {
                    blink_search::search_range::<blink_fermi_gbm::types::FermiGbm>(
                        start, end, workers, worker,
                    )
                }
                Instrument::Grid02 => blink_search::search_range::<blink_grid::types::Grid02>(
                    start, end, workers, worker,
                ),
                Instrument::Grid03b => blink_search::search_range::<blink_grid::types::Grid03B>(
                    start, end, workers, worker,
                ),
                Instrument::Grid04 => blink_search::search_range::<blink_grid::types::Grid04>(
                    start, end, workers, worker,
                ),
                Instrument::Grid07 => blink_search::search_range::<blink_grid::types::Grid07>(
                    start, end, workers, worker,
                ),
                Instrument::GecamA => blink_search::search_range::<blink_gecam::types::GecamA>(
                    start, end, workers, worker,
                ),
                Instrument::GecamB => blink_search::search_range::<blink_gecam::types::GecamB>(
                    start, end, workers, worker,
                ),
                Instrument::GecamC => blink_search::search_range::<blink_gecam::types::GecamC>(
                    start, end, workers, worker,
                ),
            }
        }
        TopCommands::Wwlln {
            instrument,
            window_ms,
        } => match instrument {
            Instrument::HxmtHe => blink_wwlln::run::<blink_hxmt_he::types::HxmtHe>(window_ms),
            Instrument::SvomGrm => blink_wwlln::run::<blink_svom_grm::types::SvomGrm>(window_ms),
            Instrument::FermiGbm => blink_wwlln::run::<blink_fermi_gbm::types::FermiGbm>(window_ms),
            Instrument::Grid02 => blink_wwlln::run::<blink_grid::types::Grid02>(window_ms),
            Instrument::Grid03b => blink_wwlln::run::<blink_grid::types::Grid03B>(window_ms),
            Instrument::Grid04 => blink_wwlln::run::<blink_grid::types::Grid04>(window_ms),
            Instrument::Grid07 => blink_wwlln::run::<blink_grid::types::Grid07>(window_ms),
            Instrument::GecamA => blink_wwlln::run::<blink_gecam::types::GecamA>(window_ms),
            Instrument::GecamB => blink_wwlln::run::<blink_gecam::types::GecamB>(window_ms),
            Instrument::GecamC => blink_wwlln::run::<blink_gecam::types::GecamC>(window_ms),
        },
        TopCommands::AcdAudit { list, out, scint } => {
            commands::acd_audit::cmd_acd_audit(&list, &out, &scint);
        }
        TopCommands::Exposure {
            from,
            to,
            force,
            workers,
            worker,
            instrument,
        } => {
            let parse = |s: &str| {
                chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .unwrap_or_else(|e| panic!("invalid date '{s}': {e}"))
            };
            let (from, to) = (parse(&from), parse(&to));
            assert!(workers >= 1 && worker < workers, "--worker out of range");
            use blink_search::exposure::exposure_range as run;
            match instrument {
                Instrument::HxmtHe => run::<blink_hxmt_he::types::HxmtHe>(from, to, workers, worker, force),
                Instrument::SvomGrm => run::<blink_svom_grm::types::SvomGrm>(from, to, workers, worker, force),
                Instrument::FermiGbm => run::<blink_fermi_gbm::types::FermiGbm>(from, to, workers, worker, force),
                Instrument::Grid02 => run::<blink_grid::types::Grid02>(from, to, workers, worker, force),
                Instrument::Grid03b => run::<blink_grid::types::Grid03B>(from, to, workers, worker, force),
                Instrument::Grid04 => run::<blink_grid::types::Grid04>(from, to, workers, worker, force),
                Instrument::Grid07 => run::<blink_grid::types::Grid07>(from, to, workers, worker, force),
                Instrument::GecamA => run::<blink_gecam::types::GecamA>(from, to, workers, worker, force),
                Instrument::GecamB => run::<blink_gecam::types::GecamB>(from, to, workers, worker, force),
                Instrument::GecamC => run::<blink_gecam::types::GecamC>(from, to, workers, worker, force),
            }
        }
        TopCommands::Vet {
            from,
            to,
            list,
            out,
            max_fa,
            force,
            workers,
            worker,
            instrument,
        } => {
            assert!(
                instrument == Instrument::HxmtHe,
                "vet: only hxmt-he is implemented, got {instrument:?}"
            );
            if let Some(list) = list {
                let out = out.expect("--list requires -o");
                commands::vet::cmd_vet_list(&list, &out);
            } else {
                let parse = |s: &str| {
                    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                        .unwrap_or_else(|e| panic!("invalid date '{s}': {e}"))
                };
                let (from, to) = (parse(&from.unwrap()), parse(&to.unwrap()));
                assert!(workers >= 1 && worker < workers, "--worker out of range");
                commands::vet::cmd_vet_days(from, to, workers, worker, max_fa, force);
            }
        }
        TopCommands::Catalog {
            input,
            out,
            out_dir,
            data,
            train_threshold,
            train_threshold_from_pool,
            settle_days,
            instrument,
        } => match (out, out_dir) {
            (Some(out), _) => commands::catalog::cmd_catalog(&input, &out),
            (None, Some(dir)) => {
                use commands::catalog_pipeline::cmd_catalog_pipeline as run;
                let t = (!train_threshold_from_pool).then_some(train_threshold);
                match instrument {
                    Instrument::HxmtHe => run::<blink_hxmt_he::types::HxmtHe>(&data, &dir, t, settle_days),
                    Instrument::SvomGrm => run::<blink_svom_grm::types::SvomGrm>(&data, &dir, t, settle_days),
                    Instrument::FermiGbm => run::<blink_fermi_gbm::types::FermiGbm>(&data, &dir, t, settle_days),
                    Instrument::Grid02 => run::<blink_grid::types::Grid02>(&data, &dir, t, settle_days),
                    Instrument::Grid03b => run::<blink_grid::types::Grid03B>(&data, &dir, t, settle_days),
                    Instrument::Grid04 => run::<blink_grid::types::Grid04>(&data, &dir, t, settle_days),
                    Instrument::Grid07 => run::<blink_grid::types::Grid07>(&data, &dir, t, settle_days),
                    Instrument::GecamA => run::<blink_gecam::types::GecamA>(&data, &dir, t, settle_days),
                    Instrument::GecamB => run::<blink_gecam::types::GecamB>(&data, &dir, t, settle_days),
                    Instrument::GecamC => run::<blink_gecam::types::GecamC>(&data, &dir, t, settle_days),
                }
            }
            (None, None) => unreachable!("clap requires -o or --out-dir"),
        },
    }
}
