#!/usr/bin/env node
/**
 * Merge Traffic Data Script
 *
 * Merges daily GitHub Traffic API snapshots into cumulative history files.
 * GitHub Traffic API only retains 14 days of data, so we accumulate daily
 * snapshots to build a long-term history.
 *
 * Output files:
 * - analytics/history/daily-views.json: Cumulative daily view counts
 * - analytics/history/daily-clones.json: Cumulative daily clone counts
 */

const fs = require('fs');
const path = require('path');

const RAW_DIR = path.join(__dirname, '..', 'raw');
const HISTORY_DIR = path.join(__dirname, '..', 'history');

// Ensure history directory exists
if (!fs.existsSync(HISTORY_DIR)) {
  fs.mkdirSync(HISTORY_DIR, { recursive: true });
}

/**
 * Load JSON file safely
 */
function loadJSON(filepath) {
  try {
    if (fs.existsSync(filepath)) {
      return JSON.parse(fs.readFileSync(filepath, 'utf8'));
    }
  } catch (e) {
    console.warn(`Warning: Could not parse ${filepath}`);
  }
  return null;
}

/**
 * Save JSON file with pretty formatting
 */
function saveJSON(filepath, data) {
  fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
}

/**
 * Merge view/clone data from raw files into history
 * @param {string} type - 'views' or 'clones'
 */
function mergeTrafficData(type) {
  const historyFile = path.join(HISTORY_DIR, `daily-${type}.json`);

  // Load existing history
  let history = loadJSON(historyFile) || { [type]: [], lastUpdated: null };

  // Create a map of existing entries by timestamp
  const existingMap = new Map();
  for (const entry of history[type]) {
    existingMap.set(entry.timestamp, entry);
  }

  // Find all raw files for this type
  const rawFiles = fs.readdirSync(RAW_DIR)
    .filter(f => f.startsWith(`${type}-`) && f.endsWith('.json'))
    .sort();

  let newEntries = 0;

  for (const filename of rawFiles) {
    const data = loadJSON(path.join(RAW_DIR, filename));
    if (!data || !data[type]) continue;

    for (const entry of data[type]) {
      // Skip if we already have this timestamp
      if (existingMap.has(entry.timestamp)) continue;

      existingMap.set(entry.timestamp, {
        timestamp: entry.timestamp,
        count: entry.count,
        uniques: entry.uniques
      });
      newEntries++;
    }
  }

  // Convert map back to sorted array
  history[type] = Array.from(existingMap.values())
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  history.lastUpdated = new Date().toISOString();

  saveJSON(historyFile, history);

  console.log(`[${type}] Merged ${newEntries} new entries. Total: ${history[type].length} days.`);
}

/**
 * Clean up old raw files (keep last 30 days)
 */
function cleanupOldRawFiles() {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const files = fs.readdirSync(RAW_DIR);
  let deleted = 0;

  for (const filename of files) {
    // Extract date from filename (e.g., views-2026-02-20.json)
    const match = filename.match(/\d{4}-\d{2}-\d{2}/);
    if (!match) continue;

    const fileDate = new Date(match[0]);
    if (fileDate < thirtyDaysAgo) {
      fs.unlinkSync(path.join(RAW_DIR, filename));
      deleted++;
    }
  }

  if (deleted > 0) {
    console.log(`Cleaned up ${deleted} old raw files.`);
  }
}

// Main execution
console.log('Merging traffic data...');
mergeTrafficData('views');
mergeTrafficData('clones');
cleanupOldRawFiles();
console.log('Done.');
