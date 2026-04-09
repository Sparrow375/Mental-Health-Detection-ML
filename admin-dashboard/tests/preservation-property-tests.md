# Preservation Property Tests - Development and Shallow Route Behavior

**Property 2: Preservation - Development and Shallow Route Behavior**

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Test Objective

These tests capture the current working behavior on UNFIXED code for non-buggy inputs (development mode and shallow routes). The goal is to ensure that when we implement the fix for nested routes, we don't break any existing functionality.

**IMPORTANT**: These tests should PASS on unfixed code to establish the baseline behavior that must be preserved.

## Testing Approach

Following the observation-first methodology:
1. Observe behavior on UNFIXED code
2. Document the observed behavior patterns
3. Write property-based tests that capture these patterns
4. Run tests on UNFIXED code to confirm they pass
5. After implementing the fix, re-run these same tests to ensure no regressions

## Test Cases

### Test Case 1: Development Server Starts and Serves Assets Correctly
**Requirement**: 3.1 - Development mode must continue to work correctly

**Property**: For any file access in development mode, the dev server SHALL serve assets correctly with hot module replacement enabled.

**Test Steps**:
1. Start development server: `npm run dev`
2. Verify server starts without errors
3. Access `http://localhost:5173/` (or configured port)
4. Verify application loads and renders
5. Open browser DevTools → Network tab
6. Verify all assets load with 200 status codes
7. Check console for any errors

**Expected Result (UNFIXED CODE)**: ✅ PASS
- Development server starts successfully
- Application loads at localhost
- All assets (JS, CSS) load with 200 status
- No console errors
- Application renders correctly

**Observed Behavior**:
- Server start time: _______________
- Port: _______________
- Asset loading: _______________
- Console status: _______________

---

### Test Case 2: Hot Module Replacement Works in Development
**Requirement**: 3.1 - Development mode with HMR must continue to work

**Property**: For any source file change in development mode, the dev server SHALL apply hot module replacement without full page reload.

**Test Steps**:
1. Start development server: `npm run dev`
2. Open application in browser
3. Open browser DevTools → Console
4. Make a small change to a React component (e.g., change text in `src/App.tsx`)
5. Save the file
6. Observe browser behavior

**Expected Result (UNFIXED CODE)**: ✅ PASS
- File change is detected by Vite
- HMR update is applied without full page reload
- Console shows HMR update message
- Component updates with new content
- Application state is preserved (if applicable)

**Observed Behavior**:
- HMR trigger time: _______________
- Console messages: _______________
- Page reload: _______________
- State preservation: _______________

---

### Test Case 3: Root Dashboard Route Loads in Production Build
**Requirement**: 3.2 - Root dashboard route must continue to load correctly

**Property**: For the shallow route `/dashboard` in production build, all assets SHALL load successfully and the application SHALL render without errors.

**Test Steps**:
1. Build the application: `npm run build`
2. Serve the production build locally: `npm run preview`
3. Navigate to `http://localhost:4173/dashboard` (or configured preview port)
4. Open browser DevTools → Network tab
5. Verify all assets load successfully
6. Check console for errors

**Expected Result (UNFIXED CODE)**: ✅ PASS
- Production build completes successfully
- Preview server starts
- `/dashboard` route loads correctly
- All assets load with 200 status
- No console errors
- Dashboard UI renders correctly

**Observed Behavior**:
- Build output: _______________
- Asset paths in HTML: _______________
- Network requests: _______________
- Rendering status: _______________

---

### Test Case 4: Login Page Loads in Production Build
**Requirement**: 3.3 - Login page must continue to load correctly

**Property**: For the shallow route `/login` in production build, all assets SHALL load successfully and the application SHALL render without errors.

**Test Steps**:
1. Build the application: `npm run build`
2. Serve the production build locally: `npm run preview`
3. Navigate to `http://localhost:4173/login`
4. Open browser DevTools → Network tab
5. Verify all assets load successfully
6. Check console for errors

**Expected Result (UNFIXED CODE)**: ✅ PASS
- `/login` route loads correctly
- All assets load with 200 status
- No console errors
- Login UI renders correctly

**Observed Behavior**:
- Asset loading: _______________
- Network requests: _______________
- Rendering status: _______________
- Authentication UI: _______________

---

### Test Case 5: Client-Side Navigation Between Dashboard Sections
**Requirement**: 3.4 - Client-side routing must continue to work correctly

**Property**: For any client-side navigation between dashboard sections, the application SHALL navigate without errors and maintain application state.

**Test Steps**:
1. Build and serve production build: `npm run build && npm run preview`
2. Navigate to `http://localhost:4173/dashboard`
3. Verify dashboard loads
4. Click navigation links to different sections (if available)
5. Use browser back/forward buttons
6. Observe navigation behavior and console

**Expected Result (UNFIXED CODE)**: ✅ PASS
- Client-side navigation works smoothly
- No page reloads during navigation
- No console errors
- Application state is maintained
- Browser history works correctly

**Observed Behavior**:
- Navigation behavior: _______________
- State preservation: _______________
- Console status: _______________
- History API: _______________

---

### Test Case 6: Build Output Structure is Consistent
**Requirement**: All requirements - Build process must remain unchanged

**Property**: For any production build, the output structure in `dist/` directory SHALL be consistent with expected patterns.

**Test Steps**:
1. Clean previous build: `rm -rf dist` (or `rmdir /s /q dist` on Windows)
2. Build the application: `npm run build`
3. Inspect `dist/` directory structure
4. Verify asset organization

**Expected Result (UNFIXED CODE)**: ✅ PASS
- `dist/` directory is created
- `dist/index.html` exists
- `dist/assets/` directory contains JS and CSS files
- Asset filenames follow pattern: `index-[hash].js` and `index-[hash].css`
- Static assets (favicon, icons) are copied to `dist/`

**Expected Directory Structure**:
```
dist/
├── index.html
├── favicon.svg
├── icons.svg
└── assets/
    ├── index-[hash].js
    └── index-[hash].css
```

**Observed Behavior**:
- Directory structure: _______________
- Asset naming pattern: _______________
- File count: _______________
- Total size: _______________

---

### Test Case 7: Asset Paths in Built HTML
**Requirement**: All requirements - Asset references must be generated correctly

**Property**: For any production build, the `index.html` SHALL contain asset references that follow a consistent pattern.

**Test Steps**:
1. Build the application: `npm run build`
2. Open `dist/index.html` in a text editor
3. Inspect `<script>` and `<link>` tags
4. Document asset path patterns

**Expected Result (UNFIXED CODE)**: ✅ PASS
- Asset paths start with `/assets/` (absolute paths)
- Script tags have `type="module"` and `crossorigin` attributes
- CSS links have `rel="stylesheet"` attribute
- No broken or malformed references

**Expected Pattern**:
```html
<script type="module" crossorigin src="/assets/index-[hash].js"></script>
<link rel="stylesheet" crossorigin href="/assets/index-[hash].css">
```

**Observed Behavior**:
- Script src pattern: _______________
- Link href pattern: _______________
- Additional attributes: _______________
- Base tag present: _______________

---

## Property-Based Test Implementation

### Test Framework

Based on the design document, we should use property-based testing to generate many test cases automatically. However, since the project doesn't currently have a PBT framework installed, we'll document the property-based approach and implement it with the available tools.

### Properties to Test

**Property 1: Development Mode Stability**
```
FOR ALL source_file_changes IN development_mode DO
  ASSERT dev_server_responds_successfully()
  ASSERT hmr_updates_without_full_reload()
  ASSERT no_console_errors()
END FOR
```

**Property 2: Shallow Route Consistency**
```
FOR ALL shallow_routes IN ["/", "/dashboard", "/login"] DO
  ASSERT assets_load_successfully(shallow_route)
  ASSERT application_renders_without_errors(shallow_route)
  ASSERT network_requests_return_200(shallow_route)
END FOR
```

**Property 3: Client-Side Navigation Preservation**
```
FOR ALL navigation_sequences IN valid_navigation_paths DO
  ASSERT navigation_completes_without_errors(navigation_sequence)
  ASSERT application_state_is_maintained(navigation_sequence)
  ASSERT no_page_reloads_occur(navigation_sequence)
END FOR
```

**Property 4: Build Output Consistency**
```
FOR ALL builds IN multiple_build_runs DO
  ASSERT build_completes_successfully()
  ASSERT output_structure_matches_expected_pattern()
  ASSERT asset_hashes_are_deterministic_for_same_content()
END FOR
```

---

## Test Execution Results

### Execution Date: _______________ (To be filled during test run)

### Test Case 1: Development Server
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 2: Hot Module Replacement
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 3: Root Dashboard Route
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 4: Login Page
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 5: Client-Side Navigation
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 6: Build Output Structure
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

### Test Case 7: Asset Paths in HTML
- Status: ⏳ Pending
- Result: _______________
- Notes: _______________

---

## Test Completion Criteria

These preservation tests are complete when:
- ✅ All test cases have been executed on UNFIXED code
- ✅ All tests PASS (confirming baseline behavior)
- ✅ Observed behaviors have been documented
- ✅ Property-based test patterns have been defined
- ✅ Test results are ready for comparison after fix implementation

## Expected Outcome

**All tests should PASS on unfixed code.** This confirms:
1. Development mode works correctly
2. Shallow routes work correctly in production
3. Client-side routing works correctly
4. Build output is consistent and correct

After implementing the fix (Task 3), we will re-run these SAME tests to ensure no regressions have been introduced.

## Next Steps

1. Execute all test cases on unfixed code
2. Document observed behaviors
3. Confirm all tests pass
4. Mark task as complete
5. Proceed to Task 3: Implement the fix
