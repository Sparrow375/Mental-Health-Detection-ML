# Bugfix Requirements Document

## Introduction

The admin dashboard React application crashes with "Minified React error #310" and multiple asset loading errors when users navigate to or directly access patient detail pages (e.g., `/dashboard/patients/Gp1UFzgJGFU6RlIE2VEr7z9x7Ay2`). The issue occurs in production on Firebase hosting but works correctly in development mode. The root cause is that Vite generates asset references with absolute paths (`/assets/index-*.js`), which fail to resolve correctly when the application is accessed via nested routes on Firebase hosting. When the browser is on a nested route and tries to load assets using absolute paths, it attempts to fetch them relative to the current URL path instead of the domain root, resulting in 404 errors and application crash.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user directly accesses a patient detail page URL (e.g., `/dashboard/patients/[patientId]`) in production THEN the system fails to load JavaScript and CSS assets from `/assets/*` paths, resulting in "Minified React error #310" and application crash

1.2 WHEN a user navigates from within the application to a patient detail page in production THEN the system fails to load assets and crashes with asset loading errors

1.3 WHEN the application is deployed to Firebase hosting without a base path configuration THEN Vite-generated asset references use absolute paths that fail to resolve on nested routes

### Expected Behavior (Correct)

2.1 WHEN a user directly accesses a patient detail page URL (e.g., `/dashboard/patients/[patientId]`) in production THEN the system SHALL successfully load all JavaScript and CSS assets and render the page without errors

2.2 WHEN a user navigates from within the application to a patient detail page in production THEN the system SHALL successfully load all required assets and render the page without crashes

2.3 WHEN the application is deployed to Firebase hosting THEN Vite SHALL generate asset references that resolve correctly from any route depth, either through proper base path configuration or relative path resolution

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the application runs in development mode (npm run dev) THEN the system SHALL CONTINUE TO load assets correctly and function without errors

3.2 WHEN a user accesses the root dashboard route (`/dashboard`) in production THEN the system SHALL CONTINUE TO load and render correctly

3.3 WHEN a user accesses the login page (`/login`) in production THEN the system SHALL CONTINUE TO load and render correctly

3.4 WHEN a user navigates between different dashboard sections (patients list, settings, etc.) THEN the system SHALL CONTINUE TO function correctly without asset loading issues
