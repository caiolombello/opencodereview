//! `ocr-bench github`: reviews one open pull request from a checkout mounted at `/work/repo` and
//! publishes the result on GitHub as the server does, then exits. It is the one-shot mode for CI
//! and for machines without the Compose stack: the same image, sandbox and pipeline as the
//! benchmark, with the server's publication code (`handle_completed_review` with
//! `auto_post_mode = comment`) run against the container's throwaway database.
//!
//! Inputs: `OCR_GITHUB_REPOSITORY` (or `GITHUB_REPOSITORY`) and `OCR_PR_NUMBER`, a GitHub token in
//! `GH_TOKEN` (or `GITHUB_TOKEN`) that may read the pull request and write reviews and comments,
//! and the provider sign-in the benchmark takes. Base and head come from GitHub; both commits must
//! be in the checkout.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::json;

use super::{Engine, PullRequest};

const DEFAULT_DEADLINE_SECS: u64 = 25 * 60;
/// The installation identity every one-shot run shares. The server keeps one per database; a
/// one-shot run starts from an empty one, so a fixed identity is what lets a later run find the
/// result comment and the review markers of an earlier run on the same pull request.
const ONE_SHOT_INSTALLATION_ID: &str = "0ce5407e-5ea1-4c0d-9e5e-000000000001";

pub fn run_from_env() -> Result<(), String> {
    let env = |name: &str| std::env::var(name).ok().filter(|value| !value.trim().is_empty());
    let nwo = env("OCR_GITHUB_REPOSITORY")
        .or_else(|| env("GITHUB_REPOSITORY"))
        .ok_or("OCR_GITHUB_REPOSITORY is required (owner/name)")?;
    let number: i64 = env("OCR_PR_NUMBER")
        .ok_or("OCR_PR_NUMBER is required")?
        .parse()
        .map_err(|_| "OCR_PR_NUMBER must be a number")?;
    if env("GH_TOKEN").or_else(|| env("GITHUB_TOKEN")).is_none() {
        return Err("GH_TOKEN is required: a token that can read the pull request and post reviews".into());
    }
    let repo = PathBuf::from(env("OCR_REPO").unwrap_or_else(|| "/work/repo".into()));

    let live = crate::github::live_pr_state(&nwo, number)?;
    if live.state != "OPEN" {
        return Err(format!(
            "{nwo}#{number} is {}; only open pull requests are reviewed",
            live.state
        ));
    }
    for (what, sha) in [("head", &live.head_sha), ("base", &live.base_sha)] {
        let found = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["cat-file", "-e", &format!("{sha}^{{commit}}")])
            .status()
            .map_err(|error| error.to_string())?
            .success();
        if !found {
            return Err(format!(
                "The {what} commit {sha} is not in the checkout at {}; fetch the full history (actions/checkout with fetch-depth: 0)",
                repo.display()
            ));
        }
    }
    let (title, body) = crate::github::pr_description(&nwo, number)?;
    let pr = PullRequest {
        nwo: nwo.clone(),
        number,
        base: live.base_sha.clone(),
        head: live.head_sha.clone(),
        title,
        body,
    };

    let engine = Engine {
        provider: env("OCR_PROVIDER").unwrap_or_else(|| "claude".into()),
        model: env("OCR_MODEL"),
        effort: env("OCR_EFFORT"),
    };
    if !matches!(engine.provider.as_str(), "claude" | "codex") {
        return Err(format!("Unsupported OCR_PROVIDER {}", engine.provider));
    }
    let deadline = Duration::from_secs(
        env("OCR_DEADLINE_SECS")
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| "OCR_DEADLINE_SECS must be a number of seconds")
            })
            .transpose()?
            .unwrap_or(DEFAULT_DEADLINE_SECS),
    );
    let credential = super::take_credential(&engine.provider)?;
    let db = crate::db::Db::from_env()?;
    db.connection()?
        .execute(
            "UPDATE settings SET value = $1 WHERE key = 'installation_id'",
            crate::pg_params![ONE_SHOT_INSTALLATION_ID],
        )
        .map_err(|error| error.to_string())?;
    super::seed_account(&db, &engine, &credential)?;
    drop(credential);
    {
        let database = db.open()?;
        // Publish findings as a comment review, as the server's automatic mode does. The PR summary
        // comment is a separate model call that the server runs in the background; a one-shot run
        // would exit before it finishes.
        database.save_setting("auto_post_mode", "comment")?;
        database.save_setting("pr_summary_comment_enabled", "false")?;
        if let Some(max_files) = env("OCR_MAX_FILES") {
            database
                .save_setting("review_max_files", &max_files)
                .map_err(|error| format!("OCR_MAX_FILES: {error}"))?;
        }
    }
    println!(
        "ocr-bench: reviewing {nwo}#{number} at {} with {}",
        &live.head_sha[..12],
        super::engine_label(&db, &engine.provider)?
    );

    let archive_dir = PathBuf::from(env("OCR_DATA_DIR").unwrap_or_else(|| "/ocr/data".into())).join("archives");
    let budget = crate::review::budget::Budget::new(deadline.saturating_sub(Duration::from_secs(120)));
    let inputs = super::prepare(&db, &repo, &pr, archive_dir, budget)?;
    // What the server's PR sync records and its publication checks read: the live base branch, so
    // the review's scope matches GitHub's when the result is published.
    let conn = db.connection()?;
    conn.execute(
        "UPDATE pull_requests SET state = 'open', base_ref = $2, base_sha = $3, url = $4 WHERE id = $1",
        crate::pg_params![
            inputs.pr_id,
            live.base_ref,
            live.base_sha,
            format!("https://github.com/{nwo}/pull/{number}")
        ],
    )
    .map_err(|error| error.to_string())?;
    drop(conn);

    let reviewed = super::run(&db, &inputs, deadline);
    let database = db.open()?;
    let output = match reviewed {
        Ok(reviewed) => reviewed.output,
        Err(error) => {
            let _ = database.fail_review_run(inputs.run_id, &error);
            return Err(format!("The review failed: {error}"));
        }
    };
    let done_phase = crate::server::completed_review_phase(&database, inputs.run_id);
    database.set_review_run_risk(inputs.run_id, &output.risk)?;
    let usage = &output.usage;
    database.set_review_run_usage(
        inputs.run_id,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_creation_tokens,
        usage.model.as_deref(),
    )?;
    if !database.finish_review_run(
        inputs.run_id,
        &output.body,
        output.comments_json.as_deref(),
        &done_phase,
    )? {
        return Err("The review run was no longer running when it finished".into());
    }
    // The offline review records the base commit as its base; publication compares it with the live
    // branch name.
    db.connection()?
        .execute(
            "UPDATE review_runs SET stacked_base_ref = $2 WHERE id = $1",
            crate::pg_params![inputs.run_id, live.base_ref],
        )
        .map_err(|error| error.to_string())?;
    let has_findings = crate::server::review_has_findings(output.comments_json.as_deref(), Some(&output.body));
    crate::server::handle_completed_review(&database, &inputs, output.no_new_changes, has_findings);
    let _ = database.finalize_review_coverage(inputs.run_id);

    let result = database.review_result(inputs.pr_id)?;
    let report = json!({
        "repository": nwo,
        "number": number,
        "head": live.head_sha,
        "findings": has_findings,
        "result": result.as_ref().map(|result| json!({
            "state": result.effective_state,
            "publication": result.publication_status,
            "reviewUrl": result.github_review_url,
            "commentId": result.github_comment_id,
            "error": result.error_message,
        })),
        "usage": {
            "inputTokens": usage.input_tokens,
            "outputTokens": usage.output_tokens,
            "cacheReadTokens": usage.cache_read_tokens,
            "cacheCreationTokens": usage.cache_creation_tokens,
        },
    });
    println!("{report}");
    if let Some(path) = env("OCR_OUT") {
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("Could not write {path}: {error}"))?;
    }
    // `disabled` is the result comment the settings leave out: the server posts one only for a clean
    // review. Findings go out as the review itself.
    match result {
        Some(result)
            if result.error_message.is_none()
                && matches!(result.publication_status.as_str(), "posted" | "disabled") =>
        {
            Ok(())
        }
        Some(result) => Err(format!(
            "The result was not published: {} ({})",
            result.publication_status,
            result.error_message.unwrap_or_default()
        )),
        None => Err("The review finished without a result to publish".into()),
    }
}
