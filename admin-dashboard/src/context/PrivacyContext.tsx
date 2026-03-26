import React, { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';

interface PrivacyContextProps {
  isAnonymous: boolean;
  togglePrivacy: () => void;
}

const PrivacyContext = createContext<PrivacyContextProps>({ 
  isAnonymous: true, 
  togglePrivacy: () => {} 
});

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

export const usePrivacy = () => useContext(PrivacyContext);
