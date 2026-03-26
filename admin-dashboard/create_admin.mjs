import { initializeApp } from 'firebase/app';
import { getAuth, createUserWithEmailAndPassword } from 'firebase/auth';

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
const auth = getAuth(app);

async function createAdmin() {
  console.log("Provisioning official Hospital Admin account...");
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, "admin@gmail.com", "admin1234");
    console.log(`Successfully created Admin account with UID: ${userCredential.user.uid}`);
  } catch(e) {
    if (e.code === 'auth/email-already-in-use') {
      console.log("Admin account already exists! Ready to use.");
    } else {
      console.error("Error creating admin account:", e);
    }
  }
  process.exit(0);
}

createAdmin();
