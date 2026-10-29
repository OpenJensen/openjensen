# Workspace guide

Select or create a project from the project menu, then open a page from the sidebar. Use the API reference in the navigation for programmatic access.

## Dashboard

Open Models or Datasets, or inspect the current resource and connection status.

## Initial setup

Connect the computer beside your SO-101, preview its camera, and work through **Connect → Motor IDs → Calibrate → Teleoperate → Record**. USB access stays on that computer; captures stay there until you add them to a project.

### Understanding an unexpected stop

An encoder reading beyond the saved calibration range can stop motor torque. The error names the joint, measured value, allowed range and distance beyond it. Support the arm, then check the physical pose and calibration before enabling control again. The reading alone cannot establish whether the cause is an incomplete calibration sweep or a hardware problem.

Open **Robot logs** to see the last stop and download recent events and readings. Logging is automatic, separate from dataset recording, and survives reconnects and companion restarts. Stops are saved separately from the rolling position history. If saving fails, the panel shows the storage error. Motor checks still run independently of logging.

Control also stops when Robot Lab is hidden or closed, after a two-second browser heartbeat loss, or on a hardware communication fault. Keep the page visible during control. The saved stop reason distinguishes these cases.

## AI teleoperation

Open **ROBOT LAB → AI teleoperation** for physical SO101 control. **Agent Control** opens with **Physical SO101** selected; choose the USB port, saved calibration and workspace camera before explicitly enabling movement. Describe the task and review the selected provider, current observation and run limits. A proposed action is separate from measured execution and verified task completion.

## Datasets

Inspect a public dataset or import local files. Open an episode to play its cameras, inspect samples, and edit labels before exporting.

## Augmentation

Select a dataset, camera and clip interval. Choose an appearance edit, review the request, and compare the returned clips before downloading.

## Fine-tune

Select compatible datasets, a model, a training method and a compute target. Review the recipe, start training, and open the saved job to follow progress or download a checkpoint.

## Distill

Choose an ACT teacher and an ACT256 student recipe. Supply observations and separate episode groups, review the recipe, and open the saved job after submission.

## Quantize

Select a saved model and an available method. Review the recipe and start the job, then open its result to download the output.

## Evaluate

Select a policy, runtime and check type. Review the recipe, start the job, and inspect its saved result.

## Run

Choose simulation, observation replay or inference. Select the runtime and inputs, review the request, and open the saved job for status and output.

## Decision Lab

Select the configured scorer, enter a state and unique criteria, and start scoring.

## Models

Open a model version to inspect its lineage or start an available operation. Use Import a model to add a policy archive.

## Cloud runs

Open an application job or a configured external run to inspect its saved status and events.

## Settings

Use Modules to manage optional features, Compute to manage workers and provider connections, and Workflow settings to edit recipes. Diagnostics starts checks for the selected project.

Open the version menu and choose **Update & restart** when an update is available. Normal macOS and Linux installations started with `jensen serve` handle updates automatically, without a separate update configuration. The app prepares the new version, backs up the workspace and restarts; a failed update restores the previous version and data. Finish active jobs and stop Robot Lab before updating.
