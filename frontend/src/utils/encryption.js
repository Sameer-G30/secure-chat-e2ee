// src/utils/encryption.js
import sodium from 'libsodium-wrappers';
import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

const firebaseConfig = {
    apiKey: "AIzaSyBl0hJLJNeqry5LCc7XIMC2glvDZfXnL0M",
    authDomain: "encrypted-chat-56ab1.firebaseapp.com",
    databaseURL: "https://encrypted-chat-56ab1-default-rtdb.firebaseio.com",
    projectId: "encrypted-chat-56ab1",
    storageBucket: "encrypted-chat-56ab1.firebasestorage.app",
    messagingSenderId: "444772450734",
    appId: "1:444772450734:web:da36e1c610acaf295f083f"
};

export const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);

// ============ INITIALIZE SODIUM ============
let sodiumReady = false;

const initSodium = async () => {
    if (!sodiumReady) {
        await sodium.ready;
        sodiumReady = true;
        console.log('✅ Sodium initialized');
    }
};

// ============ CONVERT BASE64 TO UINT8ARRAY ============
const base64ToUint8Array = (base64) => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
};

// ============ CONVERT UINT8ARRAY TO BASE64 ============
const uint8ArrayToBase64 = (uint8Array) => {
    let binaryString = '';
    for (let i = 0; i < uint8Array.length; i++) {
        binaryString += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binaryString);
};

// ============ KEY GENERATION (X25519) ============
export const generateKeypair = async () => {
    await initSodium();
    
    const keypair = sodium.crypto_box_keypair();
    
    return {
        publicKey: uint8ArrayToBase64(keypair.publicKey),
        privateKey: uint8ArrayToBase64(keypair.privateKey)
    };
};

// ============ DERIVE SHARED SECRET (ECDH) ============
export const deriveSharedSecret = async (myPrivateKey, theirPublicKey) => {
    await initSodium();
    
    try {
        const privateKey = base64ToUint8Array(myPrivateKey);
        const publicKey = base64ToUint8Array(theirPublicKey);
        
        const sharedSecret = sodium.crypto_box_beforenm(publicKey, privateKey);
        
        return uint8ArrayToBase64(sharedSecret);
    } catch (error) {
        console.error('Derive shared secret error:', error);
        return null;
    }
};

// ============ EPOCH KEY MANAGEMENT ============
const epochKeys = new Map();
const KDF_CONTEXT = "chat_sec";

export const getEpochKey = async (conversationId, epoch, masterSecret) => {
    await initSodium();
    
    const cacheKey = `${conversationId}_${epoch}`;
    
    if (epochKeys.has(cacheKey)) {
        return epochKeys.get(cacheKey);
    }
    
    try {
        const master = base64ToUint8Array(masterSecret);
        
        const subkey = sodium.crypto_kdf_derive_from_key(
            32,
            epoch,
            KDF_CONTEXT,
            master
        );
        
        const epochKey = uint8ArrayToBase64(subkey);
        epochKeys.set(cacheKey, epochKey);
        
        console.log(`✅ Epoch key ${epoch} generated for conversation ${conversationId}`);
        return epochKey;
    } catch (error) {
        console.error('Epoch key generation error:', error);
        return null;
    }
};

export const cleanupEpochKeys = (conversationId, currentEpoch) => {
    const keysToRemove = [];
    for (const [keyId, _] of epochKeys) {
        const [convId, epochStr] = keyId.split('_');
        const epoch = parseInt(epochStr);
        if (convId === conversationId && epoch < currentEpoch - 10) {
            keysToRemove.push(keyId);
        }
    }
    keysToRemove.forEach(keyId => epochKeys.delete(keyId));
    if (keysToRemove.length > 0) {
        console.log(`🧹 Cleaned up ${keysToRemove.length} old epoch keys for ${conversationId}`);
    }
};

// ============ ✅ ADD THIS FUNCTION ============
export const clearEpochKeys = (conversationId) => {
    const keysToRemove = [];
    for (const [keyId, _] of epochKeys) {
        const [convId, _epoch] = keyId.split('_');
        if (convId === conversationId) {
            keysToRemove.push(keyId);
        }
    }
    keysToRemove.forEach(keyId => epochKeys.delete(keyId));
    console.log(`🧹 Cleared all epoch keys for ${conversationId}`);
};

// ============ XChaCha20-Poly1305 ENCRYPT ============
export const encryptMessage = async (message, masterSecret, conversationId, epoch) => {
    await initSodium();
    
    try {
        const epochKey = await getEpochKey(conversationId, epoch, masterSecret);
        if (!epochKey) {
            throw new Error('Failed to derive epoch key');
        }
        
        const key = base64ToUint8Array(epochKey);
        const messageBytes = new TextEncoder().encode(message);
        
        const nonce = sodium.randombytes_buf(
            sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
        );
        
        const ciphertext = sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(
            messageBytes,
            null,
            null,
            nonce,
            key
        );
        
        return {
            ciphertext: uint8ArrayToBase64(ciphertext),
            nonce: uint8ArrayToBase64(nonce),
            epoch: epoch
        };
    } catch (error) {
        console.error('Encrypt error:', error);
        return null;
    }
};

// ============ XChaCha20-Poly1305 DECRYPT ============
export const decryptMessage = async (ciphertext, nonce, masterSecret, conversationId, epoch) => {
    await initSodium();
    
    try {
        const epochKey = await getEpochKey(conversationId, epoch, masterSecret);
        if (!epochKey) {
            throw new Error('Failed to derive epoch key');
        }
        
        const key = base64ToUint8Array(epochKey);
        const ciphertextBytes = base64ToUint8Array(ciphertext);
        const nonceBytes = base64ToUint8Array(nonce);
        
        const decrypted = sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(
            null,
            ciphertextBytes,
            null,
            nonceBytes,
            key
        );
        
        return new TextDecoder().decode(decrypted);
    } catch (error) {
        console.error('Decrypt error:', error);
        return null;
    }
};

// ============ STORE KEYS ============
export const storeKeys = (publicKey, privateKey) => {
    try {
        localStorage.setItem('secureChat_publicKey', publicKey);
        localStorage.setItem('secureChat_privateKey', privateKey);
        return true;
    } catch (error) {
        console.error('Store keys error:', error);
        return false;
    }
};

// ============ GET KEYS ============
export const getKeys = () => {
    try {
        const publicKey = localStorage.getItem('secureChat_publicKey');
        const privateKey = localStorage.getItem('secureChat_privateKey');
        
        if (publicKey && privateKey) {
            return { publicKey, privateKey };
        }
        return null;
    } catch (error) {
        console.error('Get keys error:', error);
        return null;
    }
};

// ============ HAS KEYS ============
export const hasKeys = () => {
    return !!localStorage.getItem('secureChat_privateKey');
};

// ============ DELETE KEYS ============
export const deleteKeys = () => {
    localStorage.removeItem('secureChat_publicKey');
    localStorage.removeItem('secureChat_privateKey');
};