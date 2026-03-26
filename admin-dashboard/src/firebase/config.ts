import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// MHealth Cloud connection securely wired
const firebaseConfig = {
  projectId: "mhealth-a0812",
  appId: "1:596747367379:web:430019339f9eaf3c797526",
  storageBucket: "mhealth-a0812.firebasestorage.app",
  apiKey: "AIzaSyBpQEIL2zD3LdFp4bHP00lbygzT93sl9ZE",
  authDomain: "mhealth-a0812.firebaseapp.com",
  messagingSenderId: "596747367379",
  measurementId: "G-FJQ3K1DSVN"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
