import { useContext } from 'react';
import { PrivacyContext } from '../context/PrivacyContext';

export const usePrivacy = () => useContext(PrivacyContext);
