import { createContext } from 'react';

interface PrivacyContextProps {
  isAnonymous: boolean;
  togglePrivacy: () => void;
}

export const PrivacyContext = createContext<PrivacyContextProps>({
  isAnonymous: true,
  togglePrivacy: () => {}
});
