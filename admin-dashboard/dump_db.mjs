import { initializeApp } from 'firebase/app';
import { getFirestore, collection, getDocs } from 'firebase/firestore';

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
const db = getFirestore(app);

async function checkUsers() {
  console.log("Connecting to Firestore to check 'users' collection...");
  try {
    const querySnapshot = await getDocs(collection(db, "users"));
    console.log(`\nFound ${querySnapshot.size} user(s) in the database!\n`);
    querySnapshot.forEach((doc) => {
      console.log(`--- Document ID: ${doc.id} ---`);
      console.log(JSON.stringify(doc.data(), null, 2));
    });
  } catch(e) {
    console.error("Error reading db:", e);
  }
  process.exit(0);
}

checkUsers();
