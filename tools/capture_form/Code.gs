/**
 * Google Apps Script backend for the skin-analysis calibration-cohort photo capture form.
 *
 * Deploy as a Web App (Execute as: Me, Access: Anyone with the link) and share the
 * resulting URL. Everything lands in the deploying account's own Drive -- no external
 * hosting, no third-party storage, nothing leaves Google's infrastructure.
 *
 * Client-side QC (blur/exposure) already rejected bad photos before this runs; this
 * file only persists what already passed and appends one manifest row per capture in
 * the exact column shape scripts/run_validation.py and scripts/calibrate.py expect.
 */

const ROOT_FOLDER_NAME = 'SkinAnalysis-Cohort';
const MANIFEST_SHEET_NAME = 'manifest';
const CONTACTS_SHEET_NAME = 'contacts';
const CONTACTS_HEADERS = ['subject_id', 'email', 'consented_at'];

// Matches the minimum corpus manifest columns from scripts/run_validation.py, plus
// the client-measured QC numbers so a later reviewer can see what nearly failed.
const MANIFEST_HEADERS = [
  'image', 'subject_id', 'session_id', 'skin_tone', 'device', 'lighting',
  'age_band', 'makeup_facial_hair', 'consent_ack', 'submitted_at',
  'client_laplacian_var', 'client_tenengrad', 'client_mean_luma',
  'client_clipped_low_frac', 'client_clipped_high_frac',
];

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Skin Analysis - Photo Capture')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function getOrCreateRootFolder_() {
  const it = DriveApp.getFoldersByName(ROOT_FOLDER_NAME);
  if (it.hasNext()) return it.next();
  return DriveApp.createFolder(ROOT_FOLDER_NAME);
}

function getOrCreateSubjectFolder_(root, subjectId) {
  const it = root.getFoldersByName(subjectId);
  if (it.hasNext()) return it.next();
  return root.createFolder(subjectId);
}

function getOrCreateManifestSheet_(root) {
  const files = root.getFilesByName(MANIFEST_SHEET_NAME);
  if (files.hasNext()) {
    return SpreadsheetApp.open(files.next()).getActiveSheet();
  }
  // SpreadsheetApp.create() always creates at Drive root; move it into our folder.
  const ss = SpreadsheetApp.create(MANIFEST_SHEET_NAME);
  const file = DriveApp.getFileById(ss.getId());
  root.addFile(file);
  DriveApp.getRootFolder().removeFile(file);
  const sheet = ss.getActiveSheet();
  sheet.appendRow(MANIFEST_HEADERS);
  sheet.setFrozenRows(1);
  return sheet;
}

function getOrCreateContactsSheet_(root) {
  const files = root.getFilesByName(CONTACTS_SHEET_NAME);
  if (files.hasNext()) {
    return SpreadsheetApp.open(files.next()).getActiveSheet();
  }
  const ss = SpreadsheetApp.create(CONTACTS_SHEET_NAME);
  const file = DriveApp.getFileById(ss.getId());
  root.addFile(file);
  DriveApp.getRootFolder().removeFile(file);
  const sheet = ss.getActiveSheet();
  sheet.appendRow(CONTACTS_HEADERS);
  sheet.setFrozenRows(1);
  return sheet;
}

/** Kept separate from the manifest sheet so the manifest -- which gets copied around
 *  for run_validation.py / calibrate.py -- never carries an email address. Only
 *  records a subject once; a repeat submission does not duplicate the contact row. */
function recordContactIfNew_(root, subjectId, email) {
  const sheet = getOrCreateContactsSheet_(root);
  const dataRows = sheet.getLastRow() - 1; // exclude header row
  if (dataRows > 0) {
    const existing = sheet.getRange(2, 1, dataRows, 1).getValues();
    for (let i = 0; i < existing.length; i++) {
      if (existing[i][0] === subjectId) return;
    }
  }
  sheet.appendRow([subjectId, email, new Date().toISOString()]);
}

/**
 * payload: {subjectId, email, sessionId, skinTone, device, lighting, ageBand,
 *           makeupFacialHair, consentAck, imageBase64, mimeType,
 *           laplacianVar, tenengrad, meanLuma, clippedLowFrac, clippedHighFrac}
 */
function submitCapture(payload) {
  if (!payload || !payload.consentAck) {
    throw new Error('Consent is required before a photo can be stored.');
  }
  if (!payload.subjectId || !/^[A-Za-z0-9_-]{2,40}$/.test(payload.subjectId)) {
    throw new Error('Subject ID must be 2-40 letters, digits, - or _ characters.');
  }
  if (!payload.imageBase64) {
    throw new Error('No image data received.');
  }
  if (!payload.email || payload.email.indexOf('@') === -1) {
    throw new Error('A valid email is required so a deletion request can be honoured.');
  }

  // Concurrent submissions (e.g. two people opening the link at once) can otherwise
  // both find "no folder/sheet yet" and each create their own -- splitting data
  // across duplicate SkinAnalysis-Cohort folders or manifest/contacts sheets. This
  // lock serializes the find-or-create step across simultaneous executions.
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  let root, subjectFolder, sheet;
  try {
    root = getOrCreateRootFolder_();
    subjectFolder = getOrCreateSubjectFolder_(root, payload.subjectId);
    recordContactIfNew_(root, payload.subjectId, payload.email);
    sheet = getOrCreateManifestSheet_(root);
  } finally {
    lock.releaseLock();
  }

  const bytes = Utilities.base64Decode(payload.imageBase64);
  const blob = Utilities.newBlob(bytes, payload.mimeType || 'image/jpeg');
  const stamp = Utilities.formatDate(new Date(), 'UTC', "yyyyMMdd'T'HHmmss'Z'");
  const sessionId = payload.sessionId || 'a';
  const filename = payload.subjectId + '_' + sessionId + '_' + stamp + '.jpg';
  blob.setName(filename);
  subjectFolder.createFile(blob);

  sheet.appendRow([
    filename, payload.subjectId, sessionId, payload.skinTone || '',
    payload.device || '', payload.lighting || '', payload.ageBand || '',
    payload.makeupFacialHair || '', 'yes', new Date().toISOString(),
    payload.laplacianVar || '', payload.tenengrad || '', payload.meanLuma || '',
    payload.clippedLowFrac || '', payload.clippedHighFrac || '',
  ]);

  return { ok: true, filename: filename };
}

/**
 * ONE-TIME retake-request mailer for the first cohort batch (2026-08-19). Only 6 of
 * 70 submitted photos passed the real server-side QC gate (scripts/qc_report.py) --
 * mainly blur and face-too-small/pose, which the browser form couldn't fully enforce
 * before this version. This list is the 52 subjects who need a retake AND have a
 * confirmed email in the contacts sheet.
 *
 * 4 more subjects (0101, 2292, 3789, 9121797780 -- also failed, no passing photo)
 * have NO row in the contacts sheet at all despite a photo existing. Best guess:
 * before the LockService fix was added, two near-simultaneous first-ever submissions
 * could each have found "no contacts sheet yet" and created their own -- splitting
 * contact rows across duplicate `contacts` files in Drive. Check
 * SkinAnalysis-Cohort/ for more than one file literally named "contacts" before
 * assuming these 4 people just never got recorded.
 *
 * Run this ONCE from the Apps Script editor (select sendRetakeEmails in the function
 * dropdown, click Run) after redeploying the improved form. Not wired to any trigger.
 */
function sendRetakeEmails() {
  const RETAKE_LIST = [
    ['Abhi', 'abhinayakorlakunta01@gmail.com'],
    ['Abhishek', 'p.abhishekvarma@gmail.com'],
    ['Akash', 'akash.karuturi00@gmail.com'],
    ['Anitha', 'srujankoppakula6@gmail.com'],
    ['Apoorva', 'ponnekantiapoorva@gmail.com'],
    ['ArakalaSruthi', 'sruthiarakala112@gmail.com'],
    ['Ashwini', 'ashwinigalipelly@gmail.com'],
    ['Ashwitha', 'maramashwithareddy@gmail.com'],
    ['Bhavana', 'vuppubhavana5@gmail.com'],
    ['Bhavani', 'bhavanird76@gmail.com'],
    ['Deepika', 'deepikajarapala2@gmail.com'],
    ['Divya', 'divyakorlakunta956@gmail.com'],
    ['DodlaGayathriReddy', 'gayathridodla920@gmail.com'],
    ['Dodlashravanthireddy', 'shravanthidodla12@gmail.com'],
    ['Harsha', 'palepuharsha2@gmail.com'],
    ['Harshavardhansingh', 'harshasingh594@gmail.com'],
    ['JastiRaja', 'rajajasti500@gmail.com'],
    ['KajjamNanditha', 'nanditha.kajjam@aurora.edu.in'],
    ['Keerthi', 'Keerthividam17@gmail.com'],
    ['Leela', 'leelasrinivaskorlakunta613@gmail.com'],
    ['Likith', 'likithchowdary22@gmail.com'],
    ['Madhu', 'Madhusr7476@gmail.com'],
    ['Malathi', 'malathiaila25@gmail.com'],
    ['Mounika', 'mounikakukatla68@gmail.com'],
    ['Narasimha', 'narasimha.soori40@gmail.com'],
    ['Niharika', 'niharikamotur929@gmail.com'],
    ['NitinNarayanan', 'nnair3012@gmail.com'],
    ['Padmaja', 'Impadmajadoddala@gmail.com'],
    ['Rajender', 'rajenderrajender86768@gmail.com'],
    ['Rajesh', 'rksinghk919@gmail.com'],
    ['Rakshitha', 'Rakshithav2308@gmail.com'],
    ['Ruchitha', 'ruchithapodila3@gmail.com'],
    ['Sahithi', 'thavishisahithi4@gmail.com'],
    ['Sai', 'sykosaisai@gmail.com'],
    ['SaiAbhay', 'saiabhay4321@gmail.com'],
    ['SaiSachikethanReddyMandala', 'mandalasachinreddy@gmail.com'],
    ['Sainaveena', 'reddynaveena101@gmail.com'],
    ['Sairam0430', 'amarasairam38095@gmail.com'],
    ['Salutigirishkumar123', 'salukutigirish@gmail.com'],
    ['Sandeep', 'sandeepbotla12@gmail.com'],
    ['Shahid', 'ssafridi.786@gmail.com'],
    ['Shashank', 'gshashankrajgupta@gmail.com'],
    ['Sravani', 'Sravani0866@gmail.com'],
    ['Srivalli', 'srujankoppakula@gmail.com'],
    ['Srujan16', 'srujankopppakula999@gmail.com'],
    ['Venkat', 'ksmvenkat5799@gmail.com'],
    ['Venky', 'Ponnekantivenky@gmail.com'],
    ['Vighnesh', 'kurravighnesh@gmail.com'],
    ['Vishishta', 'vishishta.gunda@aurora.edu.in'],
    ['Yadagiri', 'yadagirireddy0904@gmail.com'],
    ['karthi', 'karthikeyinigummadi@gmail.com'],
    ['shiekmubeen', 'mubeenahmed44787@gmail.com'],
  ];

  const formUrl = ScriptApp.getService().getUrl();
  const subject = 'Quick retake needed for your skin-analysis photo';
  let sent = 0;
  RETAKE_LIST.forEach(([subjectId, email]) => {
    const body =
      'Hi ' + subjectId + ',\n\n' +
      "Thanks for submitting a photo earlier! A couple of the automated checks on " +
      "our end came back borderline (usually just focus or how much of the frame " +
      "your face fills), so we need one more try to use it.\n\n" +
      "Same link as before, takes about 2 minutes:\n" + formUrl + "\n\n" +
      "Tip: hold the phone about arm's length away, look straight at the camera, " +
      "and make sure there's good even light on your face. The form will tell you " +
      "right away if the new photo looks good.\n\n" +
      "Thanks again for helping out with this.";
    try {
      MailApp.sendEmail(email, subject, body);
      sent++;
    } catch (err) {
      Logger.log('failed to email ' + subjectId + ' (' + email + '): ' + err);
    }
  });
  Logger.log('sent ' + sent + ' of ' + RETAKE_LIST.length + ' retake emails');
}
