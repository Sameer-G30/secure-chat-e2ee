// src/utils/contacts.js
import { ref, get, set, push, onChildAdded, query, orderByChild, equalTo } from 'firebase/database';
import { db } from './encryption';

// ... rest of your contacts code ...
// ============ SEARCH FOR USERS ============
export const searchUsers = async (searchTerm) => {
    try {
        console.log('🔍 Searching for:', searchTerm);
        
        const usersRef = ref(db, 'users');
        const snapshot = await get(usersRef);
        const users = snapshot.val();
        
        console.log(' All users:', users);
        
        if (!users) {
            console.log(' No users found in Firebase');
            return [];
        }
        
        const results = [];
        const searchLower = searchTerm.toLowerCase().trim();
        
        for (const [uid, userData] of Object.entries(users)) {
            const username = (userData.username || '').toLowerCase();
            
            if (username === searchLower || username.includes(searchLower)) {
                results.push({
                    uid: uid,
                    username: userData.username,
                    email: userData.email,
                    publicKey: userData.publicKey
                });
            }
        }
        
        console.log(' Search results:', results);
        return results;
        
    } catch (error) {
        console.error(' Search users error:', error);
        return [];
    }
};
    

// ============ ADD CONTACT ============
export const addContact = async (userId, contactUid, contactUsername) => {
    try {
        const contactRef = ref(db, `contacts/${userId}/${contactUid}`);
        await set(contactRef, {
            username: contactUsername,
            addedAt: Date.now()
        });
        return true;
    } catch (error) {
        console.error('Add contact error:', error);
        return false;
    }
};

// ============ GET CONTACTS ============
export const getContacts = async (userId) => {
    try {
        const contactsRef = ref(db, `contacts/${userId}`);
        const snapshot = await get(contactsRef);
        const data = snapshot.val();
        
        if (!data) return [];
        
        const contacts = [];
        for (const [uid, contactData] of Object.entries(data)) {
            contacts.push({
                uid: uid,
                username: contactData.username,
                addedAt: contactData.addedAt
            });
        }
        
        return contacts;
    } catch (error) {
        console.error('Get contacts error:', error);
        return [];
    }
};

// ============ LISTEN FOR NEW CONTACTS (Real-time) ============
export const listenForContacts = (userId, callback) => {
    const contactsRef = ref(db, `contacts/${userId}`);
    
    const unsubscribe = onChildAdded(contactsRef, (snapshot) => {
        const contactData = snapshot.val();
        const contactUid = snapshot.key;
        
        callback({
            uid: contactUid,
            username: contactData.username,
            addedAt: contactData.addedAt
        });
    });
    
    return unsubscribe;
};

// ============ GET USER BY UID ============
export const getUserByUid = async (uid) => {
    try {
        const userRef = ref(db, `users/${uid}`);
        const snapshot = await get(userRef);
        return snapshot.val();
    } catch (error) {
        console.error('Get user error:', error);
        return null;
    }
};

// ============ GET PUBLIC KEY ============
export const getPublicKey = async (uid) => {
    try {
        const userRef = ref(db, `users/${uid}/publicKey`);
        const snapshot = await get(userRef);
        return snapshot.val();
    } catch (error) {
        console.error('Get public key error:', error);
        return null;
    }
};