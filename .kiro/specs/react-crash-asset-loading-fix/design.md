# React Crash Asset Loading Fix - Bugfix Design

## Overview

The admin dashboard crashes on Firebase hosting when users access nested routes (e.g., `/dashboard/patients/[patientId]`) due to incorrect asset path resolution. Vite generates absolute asset paths (`/assets/index-*.js`) in the built `index.html`, which fail to resolve correctly on nested routes in Firebase hosting. When the browser is on a nested route like `/dashboard/patients/xyz`, it attempts to fetch `/dashboard/patients/xyz/assets/index-*.js` instead of `/assets/index-*.js`, resulting in 404 errors and React error #310.

The fix involves configuring Vite's `base` option to ensure assets are referenced relative to the domain root, not the current path. This is a minimal configuration change that ensures proper asset resolution across all route depths while preserving existing functionality.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when users access nested routes and the browser fails to load assets due to incorrect path resolution
- **Property (P)**: The desired behavior - assets should load successfully from any route depth using paths that resolve correctly relative to the domain root
- **Preservation**: Existing functionality in development mode and on shallow routes that must remain unchanged
- **Vite base**: The public base path configuration that determines how asset URLs are generated during build
- **Firebase hosting rewrites**: The configuration that routes all requests to `/index.html` for client-side routing
- **Absolute path**: A path starting with `/` that should resolve from the domain root (e.g., `/assets/file.js`)
- **Nested route**: A URL path with multiple segments (e.g., `/dashboard/patients/xyz`)

## Bug Details

### Fault Condition

The bug manifests when a user accesses the application on a nested route (2+ path segments) in production on Firebase hosting. The browser attempts to load JavaScript and CSS assets using absolute paths, but due to how browsers resolve absolute paths in the context of the current URL, the assets fail to load.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type { route: string, environment: string }
  OUTPUT: boolean
  
  RETURN input.environment == "production"
         AND input.route.split('/').filter(s => s.length > 0).length >= 2
         AND assetPathsAreAbsolute()
         AND NOT assetsLoadSuccessfully()
END FUNCTION
```

### Examples

- **Example 1**: User directly navigates to `/dashboard/patients/Gp1UFzgJGFU6RlIE2VEr7z9x7Ay2`
  - Expected: Page loads with all assets
  - Actual: Browser tries to fetch `/dashboard/patients/Gp1UFzgJGFU6RlIE2VEr7z9x7Ay2/assets/index-*.js`, gets 404, React crashes with error #310

- **Example 2**: User clicks a link to navigate to `/dashboard/patients/abc123` from within the app
  - Expected: Page loads with all assets
  - Actual: Assets fail to load, application crashes

- **Example 3**: User accesses `/dashboard` (shallow route)
  - Expected: Page loads correctly
  - Actual: Page loads correctly (no bug on shallow routes)

- **Edge Case**: User accesses `/login` (single segment route)
  - Expected: Page loads correctly
  - Actual: Page loads correctly (no bug on single segment routes)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Development mode (`npm run dev`) must continue to work correctly with hot module replacement
- Root dashboard route (`/dashboard`) must continue to load and render correctly
- Login page (`/login`) must continue to load and render correctly
- All client-side navigation between routes must continue to work correctly
- Firebase hosting rewrites must continue to route all requests to `/index.html`

**Scope:**
All functionality that currently works in development mode and on shallow routes in production should be completely unaffected by this fix. This includes:
- Development server behavior
- Hot module replacement
- Client-side routing
- Firebase authentication flow
- All existing UI components and features

## Hypothesized Root Cause

Based on the bug description and analysis of the Vite configuration, the root cause is:

1. **Missing Base Path Configuration**: The `vite.config.ts` file does not specify a `base` option, which defaults to `/` but doesn't ensure proper absolute path resolution in all contexts
   - Vite generates asset references as `/assets/index-*.js` in the built HTML
   - These absolute paths should resolve from the domain root, but browser behavior can vary

2. **Browser Path Resolution Behavior**: When the browser is on a nested route and encounters an absolute path, some browsers or hosting configurations may resolve it relative to the current path context rather than the domain root
   - This is exacerbated by Firebase hosting's rewrite rules that route all requests to `/index.html`

3. **Lack of Explicit Base URL**: Without an explicit `base: '/'` configuration, Vite may not generate asset paths that are guaranteed to resolve correctly across all deployment scenarios

4. **Potential HTML Base Tag Missing**: The built HTML may lack a `<base>` tag that would instruct the browser to resolve all relative URLs from a specific base URL

## Correctness Properties

Property 1: Fault Condition - Assets Load on Nested Routes

_For any_ route access where the route has 2 or more path segments (e.g., `/dashboard/patients/xyz`) in production, the fixed build configuration SHALL generate asset references that load successfully, allowing the application to render without crashes.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Development and Shallow Route Behavior

_For any_ environment or route that is NOT a nested production route (development mode, shallow routes like `/dashboard` or `/login`), the fixed configuration SHALL produce exactly the same behavior as the original configuration, preserving all existing functionality.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `admin-dashboard/vite.config.ts`

**Function**: Vite configuration object

**Specific Changes**:
1. **Add Explicit Base Path**: Add `base: '/'` to the Vite configuration to explicitly specify that all asset paths should be resolved from the domain root
   - This ensures Vite generates asset references that are unambiguous
   - The `/` base path is the standard for SPAs deployed at the root of a domain

2. **Verify Build Output**: After the configuration change, verify that the built `index.html` still contains absolute paths starting with `/assets/`
   - The paths should remain the same, but the explicit configuration ensures consistency

3. **Alternative: Add HTML Base Tag**: If the explicit base path doesn't resolve the issue, add a `<base href="/">` tag to the HTML head
   - This can be done via Vite's `transformIndexHtml` hook or by modifying `index.html`
   - The base tag instructs the browser to resolve all relative URLs from the specified base

4. **Test Asset Loading**: Deploy the fixed build to Firebase hosting and test asset loading on nested routes
   - Verify that `/dashboard/patients/xyz` loads assets correctly
   - Verify that browser network tab shows successful asset requests to `/assets/*`

5. **Fallback: Use Relative Paths**: If absolute paths continue to fail, configure Vite to use relative paths with `base: './'`
   - This makes all asset references relative to the HTML file location
   - May require additional testing to ensure compatibility with client-side routing

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Build the application with the current configuration, deploy to Firebase hosting, and manually test asset loading on nested routes. Use browser developer tools to observe network requests and identify the exact failure pattern.

**Test Cases**:
1. **Direct Nested Route Access**: Navigate directly to `/dashboard/patients/test123` in production (will fail on unfixed code)
   - Observe: Browser attempts to fetch assets from incorrect path
   - Expected counterexample: 404 errors for `/dashboard/patients/test123/assets/index-*.js`

2. **Client-Side Navigation to Nested Route**: Navigate from `/dashboard` to a patient detail page (will fail on unfixed code)
   - Observe: Assets fail to load after navigation
   - Expected counterexample: React error #310, asset loading failures

3. **Deep Nested Route**: Navigate to a route with 3+ segments like `/dashboard/patients/xyz/details` (will fail on unfixed code)
   - Observe: Asset loading failures at deeper nesting levels
   - Expected counterexample: Similar 404 errors with deeper path

4. **Shallow Route Test**: Navigate to `/dashboard` (should work on unfixed code)
   - Observe: Assets load correctly
   - This confirms the bug is specific to nested routes

**Expected Counterexamples**:
- Browser network tab shows 404 errors for asset requests
- Asset request URLs include the nested route path instead of resolving from domain root
- Possible causes: missing base configuration, browser path resolution behavior, Firebase hosting interaction

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL route WHERE isBugCondition({route, environment: "production"}) DO
  result := loadApplication(route)
  ASSERT assetsLoadSuccessfully(result)
  ASSERT applicationRendersWithoutErrors(result)
END FOR
```

**Test Plan**: After implementing the fix, build and deploy to Firebase hosting, then test all nested routes to verify assets load correctly.

**Test Cases**:
1. Direct access to `/dashboard/patients/[patientId]` - assets should load
2. Client-side navigation to nested routes - assets should load
3. Deep nested routes (3+ segments) - assets should load
4. Browser refresh on nested routes - assets should load

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT originalBuild(input) = fixedBuild(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for development mode and shallow routes, then write tests capturing that behavior and verify it continues after the fix.

**Test Cases**:
1. **Development Mode Preservation**: Run `npm run dev` and verify hot module replacement works correctly
2. **Shallow Route Preservation**: Access `/dashboard` and `/login` in production, verify they load correctly
3. **Client-Side Routing Preservation**: Navigate between different dashboard sections, verify routing works
4. **Build Output Preservation**: Compare built asset structure before and after fix, verify no unexpected changes

### Unit Tests

- Test that Vite configuration includes the correct `base` option
- Test that built `index.html` contains expected asset paths
- Test that asset files are generated in the correct location (`dist/assets/`)
- Test that Firebase hosting configuration remains unchanged

### Property-Based Tests

- Generate random route paths with varying depths and verify assets load correctly in production
- Generate random navigation sequences and verify application remains stable
- Test that all routes (shallow and nested) produce consistent asset loading behavior

### Integration Tests

- Test full deployment flow: build → deploy → access nested routes
- Test user authentication flow on nested routes (e.g., redirect to login, then back to patient page)
- Test that all dashboard features work correctly when accessed via nested routes
- Test browser refresh behavior on nested routes
