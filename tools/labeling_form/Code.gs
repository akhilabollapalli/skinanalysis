/**
 * Google Apps Script backend for severity labeling. Reads photos directly from the
 * SkinAnalysis-Cohort Drive folder (same folder tools/capture_form writes to) and
 * writes one row per (subject, concern, roi) to a "labels" Sheet, in exactly the
 * shape scripts/calibrate.py expects (config/calibration_gates.yaml
 * annotations.required_columns: subject_id, concern, roi, severity).
 *
 * Deploy as a Web App (Execute as: Me, Access: Anyone with the link), same as
 * tools/capture_form. Share the URL with each rater -- at least TWO independent
 * raters per concern (see tools/capture_form/NEXT_PHASE_LABELING.md), each opens
 * the same link and enters their own name/initials as their rater ID.
 */

const ROOT_FOLDER_NAME = 'SkinAnalysis-Cohort';
const LABELS_SHEET_NAME = 'labels';
const LABEL_HEADERS = ['subject_id', 'concern', 'roi', 'severity', 'rater_id', 'submitted_at'];

// Curated list of photos CONFIRMED to pass the real server-side QC gate
// (scripts/qc_report.py), not just "whatever's in the subject's Drive folder" --
// a subject folder can hold multiple attempts, most of them failed retakes. Only a
// local qc_report.py run can tell which file is the real pass, so this list is
// refreshed by hand (regenerate + redeploy) as more subjects clear the gate, the
// same pattern as tools/capture_form's retake list.
//
// Last verified: 8/8 passing, 2026-08-21.
const APPROVED = [
  { subjectId: 'Arunkalyan', filename: 'Arunkalyan_a_20260819T050148Z.jpg' },
  { subjectId: 'Kranthi', filename: 'Kranthi_a_20260819T125342Z.jpg' },
  { subjectId: 'Mohith', filename: 'Mohith_a_20260819T023758Z.jpg' },
  { subjectId: 'Sam', filename: 'Sam_a_20260819T070547Z.jpg' },
  { subjectId: 'Sivaprasad', filename: 'Sivaprasad_a_20260819T043519Z.jpg' },
  { subjectId: 'Thrived', filename: 'Thrived_a_20260819T041219Z.jpg' },
  { subjectId: 'Venkat', filename: 'Venkat_a_20260821T014132Z.jpg' },
  { subjectId: 'Divya', filename: 'Divya_a_20260821T014159Z.jpg' },
];

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Skin Analysis - Severity Labeling')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function getRootFolder_() {
  const it = DriveApp.getFoldersByName(ROOT_FOLDER_NAME);
  if (it.hasNext()) return it.next();
  throw new Error(ROOT_FOLDER_NAME + ' not found in Drive -- run the capture form first.');
}

function getOrCreateLabelsSheet_(root) {
  const files = root.getFilesByName(LABELS_SHEET_NAME);
  if (files.hasNext()) {
    return SpreadsheetApp.open(files.next()).getActiveSheet();
  }
  const ss = SpreadsheetApp.create(LABELS_SHEET_NAME);
  const file = DriveApp.getFileById(ss.getId());
  root.addFile(file);
  DriveApp.getRootFolder().removeFile(file);
  const sheet = ss.getActiveSheet();
  sheet.appendRow(LABEL_HEADERS);
  sheet.setFrozenRows(1);
  return sheet;
}

/** subject_ids this rater has already fully submitted a label set for. */
function labeledSubjectsFor_(sheet, raterId) {
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return new Set();
  const rows = sheet.getRange(2, 1, lastRow - 1, 5).getValues(); // subject_id..rater_id
  const done = new Set();
  rows.forEach((row) => {
    const [subjectId, , , , rowRaterId] = row;
    if (rowRaterId === raterId) done.add(subjectId);
  });
  return done;
}

function getNextSubject(raterId) {
  const root = getRootFolder_();
  const sheet = getOrCreateLabelsSheet_(root);
  const done = labeledSubjectsFor_(sheet, raterId);
  const next = APPROVED.find((entry) => !done.has(entry.subjectId));
  return next || null;
}

function getProgress(raterId) {
  const root = getRootFolder_();
  const sheet = getOrCreateLabelsSheet_(root);
  const done = labeledSubjectsFor_(sheet, raterId);
  return { done: done.size, total: APPROVED.length };
}

function getPhotoDataUrl(subjectId) {
  const entry = APPROVED.find((e) => e.subjectId === subjectId);
  if (!entry) throw new Error('unknown subject: ' + subjectId);
  const root = getRootFolder_();
  const subjectFolders = root.getFoldersByName(subjectId);
  if (!subjectFolders.hasNext()) throw new Error('no folder for subject: ' + subjectId);
  const subjectFolder = subjectFolders.next();
  const files = subjectFolder.getFilesByName(entry.filename);
  if (!files.hasNext()) throw new Error('photo not found: ' + entry.filename);
  const blob = files.next().getBlob();
  return 'data:' + blob.getContentType() + ';base64,' + Utilities.base64Encode(blob.getBytes());
}

/**
 * payload: {raterId, subjectId, labels: [{concern, roi, severity}, ...]}
 * One call writes the WHOLE label set for one subject -- a partial submit would
 * leave that subject looking "labeled" (any row present) while some (concern, roi)
 * pairs silently never got a judgement, which calibrate.py has no way to detect.
 */
function submitLabels(payload) {
  if (!payload || !payload.raterId || !payload.subjectId) {
    throw new Error('missing raterId or subjectId');
  }
  if (!APPROVED.some((e) => e.subjectId === payload.subjectId)) {
    throw new Error('subject not in the approved list: ' + payload.subjectId);
  }
  const labels = payload.labels || [];
  if (labels.length === 0) {
    throw new Error('no labels submitted');
  }
  const validSeverities = new Set(['not_detected', 'mild', 'moderate', 'high', 'unsure']);
  labels.forEach((l) => {
    if (!l.concern || !l.roi || !validSeverities.has(l.severity)) {
      throw new Error('invalid label row: ' + JSON.stringify(l));
    }
  });

  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const root = getRootFolder_();
    const sheet = getOrCreateLabelsSheet_(root);
    const now = new Date().toISOString();
    labels.forEach((l) => {
      sheet.appendRow([payload.subjectId, l.concern, l.roi, l.severity, payload.raterId, now]);
    });
  } finally {
    lock.releaseLock();
  }
  return { ok: true };
}
