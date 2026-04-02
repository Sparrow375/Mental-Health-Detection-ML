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

const uid = "Gg1UDzgLGEUG9IL62V5JrZ9a7Ay2";

async function checkResults() {
  // Check every possible results collection name
  for (const sub of ['results', 'analysis_results', 'ml_results', 'analysis', 'predictions', 'scores', 'anomaly_results', 'daily_results']) {
    const snap = await getDocs(collection(db, `users/${uid}/${sub}`));
    if (snap.size > 0) {
      console.log(`*** ${sub}: ${snap.size} docs ***`);
      snap.forEach(d => console.log(`  ${d.id}:`, JSON.stringify(d.data())));
    }
  }

  // Also check top-level collections that might hold results
  for (const topColl of ['results', 'analysis_results', 'ml_results', 'predictions']) {
    try {
      const snap = await getDocs(collection(db, topColl));
      if (snap.size > 0) {
        console.log(`\nTOP-LEVEL ${topColl}: ${snap.size} docs`);
        snap.docs.slice(0, 3).forEach(d => console.log(`  ${d.id}:`, JSON.stringify(d.data()).substring(0, 200)));
      }
    } catch(e) {}
  }

  process.exit(0);
}

checkResults().catch(e => { console.error(e); process.exit(1); });
