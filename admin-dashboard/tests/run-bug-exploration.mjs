#!/usr/bin/env node

/**
 * Bug Condition Exploration Test Runner
 * 
 * This script helps automate the bug exploration process by:
 * 1. Building the application with current (unfixed) configuration
 * 2. Analyzing the built assets
 * 3. Providing deployment instructions
 * 4. Generating test URLs for manual verification
 * 
 * **CRITICAL**: This test is EXPECTED TO FAIL on unfixed code
 */

import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

console.log('🔍 Bug Condition Exploration Test - Asset Loading on Nested Routes\n');
console.log('=' .repeat(70));
console.log('Property 1: Fault Condition - Assets Load on Nested Routes');
console.log('Validates: Requirements 2.1, 2.2, 2.3');
console.log('=' .repeat(70));
console.log('\n⚠️  CRITICAL: This test is EXPECTED TO FAIL on unfixed code\n');

// Step 1: Build the application
console.log('📦 Step 1: Building application with current (unfixed) configuration...\n');

try {
  execSync('npm run build', { 
    cwd: projectRoot, 
    stdio: 'inherit' 
  });
  console.log('\n✅ Build completed successfully\n');
} catch (error) {
  console.error('❌ Build failed:', error.message);
  process.exit(1);
}

// Step 2: Analyze built assets
console.log('🔬 Step 2: Analyzing built assets...\n');

const distPath = join(projectRoot, 'dist');
const indexPath = join(distPath, 'index.html');

if (!existsSync(indexPath)) {
  console.error('❌ Built index.html not found at:', indexPath);
  process.exit(1);
}

const indexHtml = readFileSync(indexPath, 'utf-8');

// Extract asset references
const scriptMatches = [...indexHtml.matchAll(/<script[^>]*src="([^"]+)"[^>]*>/g)];
const linkMatches = [...indexHtml.matchAll(/<link[^>]*href="([^"]+)"[^>]*>/g)];

console.log('📄 Asset references found in index.html:');
console.log('\nJavaScript files:');
scriptMatches.forEach(match => {
  const src = match[1];
  console.log(`  - ${src}`);
  if (src.startsWith('/assets/')) {
    console.log('    ⚠️  Absolute path - may fail on nested routes');
  } else if (src.startsWith('./assets/')) {
    console.log('    ✅ Relative path - should work on nested routes');
  }
});

console.log('\nCSS files:');
linkMatches.forEach(match => {
  const href = match[1];
  if (href.includes('.css')) {
    console.log(`  - ${href}`);
    if (href.startsWith('/assets/')) {
      console.log('    ⚠️  Absolute path - may fail on nested routes');
    } else if (href.startsWith('./assets/')) {
      console.log('    ✅ Relative path - should work on nested routes');
    }
  }
});

// Check for base tag
const baseTagMatch = indexHtml.match(/<base[^>]*href="([^"]+)"[^>]*>/);
if (baseTagMatch) {
  console.log(`\n<base> tag found: href="${baseTagMatch[1]}"`);
} else {
  console.log('\n⚠️  No <base> tag found in HTML');
}

// Step 3: Check Vite configuration
console.log('\n🔧 Step 3: Checking Vite configuration...\n');

const viteConfigPath = join(projectRoot, 'vite.config.ts');
const viteConfig = readFileSync(viteConfigPath, 'utf-8');

if (viteConfig.includes('base:')) {
  console.log('✅ Vite config has "base" option configured');
  const baseMatch = viteConfig.match(/base:\s*['"]([^'"]+)['"]/);
  if (baseMatch) {
    console.log(`   Value: "${baseMatch[1]}"`);
  }
} else {
  console.log('⚠️  Vite config does NOT have "base" option configured');
  console.log('   This is the suspected root cause of the bug');
}

// Step 4: Deployment instructions
console.log('\n' + '='.repeat(70));
console.log('📤 Step 4: Deploy to Firebase Hosting');
console.log('='.repeat(70));
console.log('\nRun the following command to deploy:');
console.log('\n  firebase deploy --only hosting\n');

// Step 5: Test instructions
console.log('='.repeat(70));
console.log('🧪 Step 5: Manual Testing Instructions');
console.log('='.repeat(70));
console.log('\nAfter deployment, test the following URLs:\n');

console.log('Test Case 1: Direct Nested Route Access');
console.log('  URL: https://[your-firebase-domain].web.app/dashboard/patients/test123');
console.log('  Expected: ❌ Assets fail to load (404 errors)\n');

console.log('Test Case 2: Client-Side Navigation');
console.log('  1. Navigate to: https://[your-firebase-domain].web.app/dashboard');
console.log('  2. Click on a patient to navigate to detail page');
console.log('  Expected: ❌ Assets fail to load after navigation\n');

console.log('Test Case 3: Shallow Route (Control)');
console.log('  URL: https://[your-firebase-domain].web.app/dashboard');
console.log('  Expected: ✅ Assets load correctly (confirms bug is route-specific)\n');

console.log('Test Case 4: Browser Refresh on Nested Route');
console.log('  1. Navigate to a patient detail page via client-side navigation');
console.log('  2. Press F5 to refresh');
console.log('  Expected: ❌ Assets fail to load on refresh\n');

// Step 6: Documentation
console.log('='.repeat(70));
console.log('📝 Step 6: Document Counterexamples');
console.log('='.repeat(70));
console.log('\nFor each failing test case, document:');
console.log('  - Asset request URLs (from Network tab)');
console.log('  - HTTP status codes (should be 404)');
console.log('  - Error messages (React error #310)');
console.log('  - Screenshots of Network tab and Console\n');

console.log('Update the test results in:');
console.log('  admin-dashboard/tests/bug-condition-exploration.md\n');

// Summary
console.log('='.repeat(70));
console.log('📊 Summary');
console.log('='.repeat(70));
console.log('\n✅ Build completed');
console.log('✅ Asset analysis completed');
console.log('⏳ Awaiting deployment and manual testing');
console.log('\n⚠️  Remember: Test failures on unfixed code are EXPECTED and CORRECT');
console.log('   They confirm the bug exists and validate our hypothesis.\n');

console.log('Next steps:');
console.log('  1. Deploy to Firebase hosting');
console.log('  2. Run manual tests in browser with DevTools open');
console.log('  3. Document counterexamples in bug-condition-exploration.md');
console.log('  4. Mark task as complete when counterexamples are documented\n');
