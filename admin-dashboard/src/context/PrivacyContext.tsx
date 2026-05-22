import React, { useState } from 'react';
import type { ReactNode } from 'react';
import { PrivacyContext } from './PrivacyContext';

export const PrivacyProvider: React.FC<{children: ReactNode}> = ({ children }) => {
  // Default to true for maximum HIPAA/privacy compliance by default
  const [isAnonymous, setIsAnonymous] = useState(true);

  return (
    <PrivacyContext.Provider value={{
      isAnonymous,
      togglePrivacy: () => setIsAnonymous(!isAnonymous)
    }}>
      {children}
    </PrivacyContext.Provider>
  );
};
