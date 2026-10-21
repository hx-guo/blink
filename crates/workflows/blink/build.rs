//! 把构建时的 git 提交写进 `BLINK_GIT`，目录的 manifest 用它记录是哪版代码产出的。
//! 工作区有未提交改动时加 `-dirty`；拿不到 git 时为 `unknown`。

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    let version = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(hash) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            if dirty { format!("{hash}-dirty") } else { hash }
        }
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=BLINK_GIT={version}");
    // 提交、暂存与源码任何改动都要重新取
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/index");
    println!("cargo:rerun-if-changed=../../../crates");
}
