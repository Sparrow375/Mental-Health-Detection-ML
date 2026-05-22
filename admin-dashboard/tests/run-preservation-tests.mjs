#!/usr/bin/env node

/**
 * Preservation Property Tests Runner
 * 
 * This script automates the preservation testing process by:
 * 1. Testing development server functionality
 * 2. Building and analyzing production output
 * 3. Verifying build output structure
 * 4. Documenting baseline behavior to preserve
 * 
 * **IMPORTANT**: These tests should PASS on unfixed code
 */

import { execSync, spawn } from 'child_process';
import { readFileSync, existsSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

console.log('🛡️  Preservation Property Tests - Development and Shallow Route Behavior\n');
console.log('=' .repeat(70));
console.log('Property 2: Preservation - Development and Shallow Route Behavior');
console.log('Validates: Requirements 3.1, 3.2, 3.3, 3.4');
console.log('=' .repeat(70));
console.log('\n✅ IMPORTANT: These tests should PASS on unfixed code\n');

let testResults = {
  passed: 0,
  failed: 0,
  total: 0
};

function logTest(name, status, details = '') {
  testResults.total++;
  if (status === 'PASS') {
    testResults.passed++;
    console.log(`✅ ${name}: PASS`);
  } else {
    testResults.failed++;
    console.log(`❌ ${name}: FAIL`);
  }
  if (details) {
    console.log(`   ${details}`);
  }
  console.log();
}

// Test Case 6: Build Output Structure (run first to have build ready)
console.log('📦 Test Case 6: Build Output Structure\n');
console.log('Building application...\n');

try {
  execSync('npm run build', { 
    cwd: projectRoot, 
    stdio: 'inherit' 
  });
  console.log('\n✅ Build completed successfully\n');
} catch (error) {
  console.error('❌ Build failed:', error.message);
  logTest('Build Output Structure', 'FAIL', 'Build process failed');
  process.exit(1);
}

// Verify build output structure
const distPath = join(projectRoot, 'dist');
const indexPath = join(distPath, 'index.html');
const assetsPath = join(distPath, 'assets');

console.log('Verifying build output structure...\n');

let structureValid = true;
let structureDetails = [];

if (!existsSync(distPath)) {
  structureValid = false;
  structureDetails.push('dist/ directory not found');
} else {
  structureDetails.push('dist/ directory exists');
}

if (!existsSync(indexPath)) {
  structureValid = false;
  structureDetails.push('dist/index.html not found');
} else {
  structureDetails.push('dist/index.html exists');
}

if (!existsSync(assetsPath)) {
  structureValid = false;
  structureDetails.push('dist/assets/ directory not found');
} else {
  const assetFiles = readdirSync(assetsPath);
  const jsFiles = assetFiles.filter(f => f.endsWith('.js'));
  const cssFiles = assetFiles.filter(f => f.endsWith('.css'));
  
  structureDetails.push(`dist/assets/ contains ${jsFiles.length} JS file(s)`);
  structureDetails.push(`dist/assets/ contains ${cssFiles.length} CSS file(s)`);
  
  if (jsFiles.length === 0) {
    structureValid = false;
    structureDetails.push('No JS files found in assets/');
  }
  
  if (cssFiles.length === 0) {
    structureValid = false;
    structureDetails.push('No CSS files found in assets/');
  }
}

// Check for static assets
const faviconPath = join(distPath, 'favicon.svg');
const iconsPath = join(distPath, 'icons.svg');

if (existsSync(faviconPath)) {
  structureDetails.push('favicon.svg copied to dist/');
}

if (existsSync(iconsPath)) {
  structureDetails.push('icons.svg copied to dist/');
}

console.log('Build output structure:');
structureDetails.forEach(detail => console.log(`  - ${detail}`));
console.log();

logTest(
  'Build Output Structure', 
  structureValid ? 'PASS' : 'FAIL',
  structureValid ? 'All expected files and directories present' : 'Missing expected files or directories'
);

// Test Case 7: Asset Paths in Built HTML
console.log('📄 Test Case 7: Asset Paths in Built HTML\n');

const indexHtml = readFileSync(indexPath, 'utf-8');

// Extract asset references
const scriptMatches = [...indexHtml.matchAll(/<script[^>]*src="([^"]+)"[^>]*>/g)];
const linkMatches = [...indexHtml.matchAll(/<link[^>]*href="([^"]+)"[^>]*>/g)];

console.log('Asset references in index.html:\n');

console.log('JavaScript files:');
let jsPathsValid = true;
scriptMatches.forEach(match => {
  const src = match[1];
  console.log(`  - ${src}`);
  
  if (src.startsWith('/assets/') && src.match(/index-[a-zA-Z0-9]+\.js/)) {
    console.log('    ✅ Valid absolute path with hash');
  } else if (!src.startsWith('http')) {
    jsPathsValid = false;
    console.log('    ⚠️  Unexpected path pattern');
  }
});

console.log('\nCSS files:');
let cssPathsValid = true;
linkMatches.forEach(match => {
  const href = match[1];
  if (href.includes('.css')) {
    console.log(`  - ${href}`);
    
    if (href.startsWith('/assets/') && href.match(/index-[a-zA-Z0-9]+\.css/)) {
      console.log('    ✅ Valid absolute path with hash');
    } else {
      cssPathsValid = false;
      console.log('    ⚠️  Unexpected path pattern');
    }
  }
});

// Check for base tag
const baseTagMatch = indexHtml.match(/<base[^>]*href="([^"]+)"[^>]*>/);
console.log();
if (baseTagMatch) {
  console.log(`<base> tag found: href="${baseTagMatch[1]}"`);
} else {
  console.log('No <base> tag in HTML (expected for unfixed code)');
}
console.log();

const assetPathsValid = jsPathsValid && cssPathsValid;
logTest(
  'Asset Paths in Built HTML',
  assetPathsValid ? 'PASS' : 'FAIL',
  assetPathsValid ? 'Asset paths follow expected pattern' : 'Asset paths have unexpected patterns'
);

// Test Case 1: Development Server (informational - requires manual verification)
console.log('🔧 Test Case 1: Development Server\n');
console.log('This test requires manual verification:\n');
console.log('Steps:');
console.log('  1. Run: npm run dev');
console.log('  2. Open browser to http://localhost:5173/');
console.log('  3. Verify application loads without errors');
console.log('  4. Check browser DevTools → Network tab for 200 status codes');
console.log('  5. Check Console for any errors\n');
console.log('Expected: ✅ Dev server starts, application loads, no errors\n');
console.log('⏭️  Skipping automated test (requires manual verification)\n');

// Test Case 2: Hot Module Replacement (informational)
console.log('🔥 Test Case 2: Hot Module Replacement\n');
console.log('This test requires manual verification:\n');
console.log('Steps:');
console.log('  1. Run: npm run dev');
console.log('  2. Open application in browser');
console.log('  3. Make a small change to src/App.tsx');
console.log('  4. Save the file');
console.log('  5. Observe browser updates without full reload\n');
console.log('Expected: ✅ HMR applies changes without full page reload\n');
console.log('⏭️  Skipping automated test (requires manual verification)\n');

// Test Case 3, 4, 5: Production routes (informational - requires preview server)
console.log('🌐 Test Cases 3, 4, 5: Production Routes and Navigation\n');
console.log('These tests require manual verification with preview server:\n');
console.log('Steps:');
console.log('  1. Run: npm run preview');
console.log('  2. Test Case 3: Navigate to http://localhost:4173/dashboard');
console.log('     Expected: ✅ Dashboard loads, all assets load with 200 status');
console.log('  3. Test Case 4: Navigate to http://localhost:4173/login');
console.log('     Expected: ✅ Login page loads, all assets load with 200 status');
console.log('  4. Test Case 5: Click navigation links between sections');
console.log('     Expected: ✅ Client-side navigation works without page reloads\n');
console.log('⏭️  Skipping automated tests (require preview server and browser)\n');

// Vite Configuration Analysis
console.log('🔧 Vite Configuration Analysis\n');

const viteConfigPath = join(projectRoot, 'vite.config.ts');
const viteConfig = readFileSync(viteConfigPath, 'utf-8');

console.log('Current Vite configuration:\n');
console.log(viteConfig);
console.log();

if (viteConfig.includes('base:')) {
  console.log('⚠️  Vite config has "base" option configured');
  const baseMatch = viteConfig.match(/base:\s*['"]([^'"]+)['"]/);
  if (baseMatch) {
    console.log(`   Value: "${baseMatch[1]}"`);
  }
  console.log('   This may indicate the fix has already been applied');
} else {
  console.log('✅ Vite config does NOT have "base" option (expected for unfixed code)');
  console.log('   This confirms we are testing the baseline configuration');
}
console.log();

// Summary
console.log('='.repeat(70));
console.log('📊 Test Summary');
console.log('='.repeat(70));
console.log();
console.log(`Total automated tests: ${testResults.total}`);
console.log(`Passed: ${testResults.passed}`);
console.log(`Failed: ${testResults.failed}`);
console.log();

if (testResults.failed === 0) {
  console.log('✅ All automated preservation tests PASSED');
  console.log();
  console.log('This confirms the baseline behavior on unfixed code:');
  console.log('  ✅ Build process works correctly');
  console.log('  ✅ Build output structure is correct');
  console.log('  ✅ Asset paths follow expected pattern');
  console.log();
  console.log('Manual verification required for:');
  console.log('  ⏳ Development server functionality');
  console.log('  ⏳ Hot module replacement');
  console.log('  ⏳ Production routes (dashboard, login)');
  console.log('  ⏳ Client-side navigation');
  console.log();
  console.log('Next steps:');
  console.log('  1. Manually verify development server and HMR (optional)');
  console.log('  2. Manually verify production routes with preview server (optional)');
  console.log('  3. Mark Task 2 as complete');
  console.log('  4. Proceed to Task 3: Implement the fix');
  console.log();
  console.log('After implementing the fix, re-run this script to ensure no regressions.');
} else {
  console.log('❌ Some preservation tests FAILED');
  console.log();
  console.log('This may indicate:');
  console.log('  - Build configuration issues');
  console.log('  - Missing dependencies');
  console.log('  - Unexpected project structure');
  console.log();
  console.log('Review the failed tests above and resolve issues before proceeding.');
}
console.log();

process.exit(testResults.failed === 0 ? 0 : 1);
