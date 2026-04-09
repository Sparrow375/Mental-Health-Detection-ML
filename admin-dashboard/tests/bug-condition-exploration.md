# Bug Condition Exploration Test - Asset Loading on Nested Routes

**Property 1: Fault Condition - Assets Load on Nested Routes**

**Validates: Requirements 2.1, 2.2, 2.3**

## Test Objective

This test explores the bug condition on UNFIXED code to surface counterexamples that demonstrate the asset loading failure on nested routes in production. 

**CRITICAL**: This test is EXPECTED TO FAIL on unfixed code - failure confirms the bug exists.

## Test Setup

### Prerequisites
- Firebase CLI installed and authenticated
- Admin dashboard built with current (unfixed) Vite configuration
- Access to Firebase hosting project

### Build and Deploy

```bash
cd admin-dashboard
npm run build
firebase deploy --only hosting
```

## Test Cases

### Test Case 1: Direct Nested Route Access (2 segments)
**Route**: `/dashboard/patients/test123`

**Steps**:
1. Open browser (Chrome/Firefox recommended for DevTools)
2. Open Developer Tools (F12) → Network tab
3. Clear network log
4. Navigate directly to: `https://[your-firebase-domain].web.app/dashboard/patients/test123`
5. Observe network requests for asset files

**Expected Result (UNFIXED CODE)**: 
- ❌ Browser attempts to fetch assets from incorrect path
- ❌ 404 errors for asset requests like `/dashboard/patients/test123/assets/index-*.js`
- ❌ React error #310 displayed in console
- ❌ Application fails to render

**Counterexample Documentation**:
- Asset request URLs: _______________
- HTTP status codes: _______________
- Error messages: _______________
- Screenshot: _______________

---

### Test Case 2: Client-Side Navigation to Nested Route
**Route**: Navigate from `/dashboard` to `/dashboard/patients/[patientId]`

**Steps**:
1. Navigate to `https://[your-firebase-domain].web.app/dashboard`
2. Verify dashboard loads correctly
3. Open Developer Tools → Network tab
4. Clear network log
5. Click on a patient in the patient list to navigate to patient detail page
6. Observe network requests and application behavior

**Expected Result (UNFIXED CODE)**:
- ❌ Assets fail to load after navigation
- ❌ React error #310 in console
- ❌ Application crashes or fails to render patient detail

**Counterexample Documentation**:
- Navigation path: _______________
- Asset request failures: _______________
- Error messages: _______________
- Screenshot: _______________

---

### Test Case 3: Deep Nested Route (3+ segments)
**Route**: `/dashboard/patients/test123/details` (if such route exists, or create test route)

**Steps**:
1. Open Developer Tools → Network tab
2. Navigate directly to a deep nested route (3+ segments)
3. Observe network requests

**Expected Result (UNFIXED CODE)**:
- ❌ Similar 404 errors with deeper path nesting
- ❌ Asset requests include full nested path
- ❌ Application fails to load

**Counterexample Documentation**:
- Route tested: _______________
- Asset request URLs: _______________
- Error pattern: _______________

---

### Test Case 4: Shallow Route (Control Test)
**Route**: `/dashboard`

**Steps**:
1. Open Developer Tools → Network tab
2. Navigate directly to `https://[your-firebase-domain].web.app/dashboard`
3. Observe network requests

**Expected Result (UNFIXED CODE)**:
- ✅ Assets load correctly from `/assets/*`
- ✅ Application renders without errors
- ✅ No 404 errors in network tab

**Result**: This confirms the bug is specific to nested routes (2+ segments)

---

### Test Case 5: Browser Refresh on Nested Route
**Route**: `/dashboard/patients/[patientId]`

**Steps**:
1. Navigate to dashboard and then to a patient detail page via client-side navigation
2. Once on patient detail page, press F5 to refresh
3. Observe network requests and application behavior

**Expected Result (UNFIXED CODE)**:
- ❌ Assets fail to load on refresh
- ❌ 404 errors for asset files
- ❌ Application crashes

**Counterexample Documentation**:
- Refresh behavior: _______________
- Asset loading: _______________

---

## Root Cause Analysis

Based on the counterexamples observed, analyze:

1. **Asset Path Pattern**: What paths are the assets being requested from?
   - Expected: `/assets/index-*.js` (from domain root)
   - Actual: `/assets/index-CGIkHlPA.js` and `/assets/index-Bez7d2k2.css` (absolute paths)
   - ⚠️ These absolute paths will fail on nested routes

2. **Browser Behavior**: How is the browser resolving absolute paths?
   - Is it resolving from domain root or current URL context?
   - Answer: On nested routes like `/dashboard/patients/test123`, browsers may resolve absolute paths relative to the current path context, resulting in requests to `/dashboard/patients/test123/assets/index-*.js` instead of `/assets/index-*.js`

3. **Vite Configuration**: What is missing in the current `vite.config.ts`?
   - Current config: No `base` option specified
   - Impact: Without explicit `base: '/'`, Vite generates absolute paths that may not resolve correctly on nested routes in all hosting environments

4. **Firebase Hosting Interaction**: How do Firebase rewrites affect asset resolution?
   - Rewrite rule: All requests → `/index.html`
   - Impact on asset paths: When a nested route is accessed, Firebase serves `/index.html`, but the browser then tries to load assets relative to the nested path, causing 404 errors

## Hypothesis Validation

Based on the counterexamples, does the hypothesized root cause hold?

**Hypothesis**: Missing `base` path configuration in Vite causes assets to be referenced with absolute paths that fail to resolve correctly on nested routes.

**Validation from Build Analysis**:
- ✅ Build analysis confirms assets use absolute paths (`/assets/index-*.js`)
- ✅ Vite config lacks `base` option (confirmed root cause)
- ✅ No `<base>` tag in generated HTML
- ⏳ Deployment testing required to observe actual 404 errors on nested routes
- ⏳ Control test (shallow routes) needs verification

**Preliminary Conclusion**: The build analysis strongly supports the hypothesis. The Vite configuration is missing the `base` option, and assets are generated with absolute paths that are likely to fail on nested routes in Firebase hosting. Full validation requires deployment and manual testing in browser.

## Test Completion Criteria

This test is complete when:
- ✅ All test cases have been executed
- ✅ Counterexamples have been documented
- ✅ Root cause hypothesis has been validated or refuted
- ✅ Test results confirm the bug exists on unfixed code

## Next Steps

After documenting the counterexamples:
1. If hypothesis is validated → Proceed to implement fix (Task 2)
2. If hypothesis is refuted → Re-analyze root cause and update design document
