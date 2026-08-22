// src/utils/messaging.js
import { ref, push, onChildAdded, query, limitToLast, get, update, remove } from 'firebase/database';
import { db } from './encryption';
import { 
    getKeys, 
    deriveSharedSecret, 
    encryptMessage, 
    decryptMessage,
    cleanupEpochKeys,
    clearEpochKeys
} from './encryption';
import { getPublicKey } from './contacts';

// ============ CONVERSATION STATE ============
const conversationEpochs = new Map();
const MESSAGES_PER_EPOCH = 50;

const getCurrentEpoch = (conversationId) => {
    if (!conversationEpochs.has(conversationId)) {
        conversationEpochs.set(conversationId, 1);
    }
    return conversationEpochs.get(conversationId);
};

const incrementEpoch = (conversationId) => {
    const current = getCurrentEpoch(conversationId);
    const next = current + 1;
    conversationEpochs.set(conversationId, next);
    console.log(`📅 Epoch advanced: ${current} → ${next} for ${conversationId}`);
    return next;
};

// ============ SEND MESSAGE ============
export const sendMessage = async (senderUid, receiverUid, messageText) => {
    try {
        const myKeys = getKeys();
        if (!myKeys) {
            throw new Error('Encryption keys not found');
        }
        
        const receiverPublicKey = await getPublicKey(receiverUid);
        if (!receiverPublicKey) {
            throw new Error('Receiver has no public key');
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            receiverPublicKey
        );
        if (!masterSecret) {
            throw new Error('Failed to derive shared secret');
        }
        
        const conversationId = [senderUid, receiverUid].sort().join('_');
        
        let currentEpoch = getCurrentEpoch(conversationId);
        
        const messagesRef = ref(db, `messages/${conversationId}`);
        const snapshot = await get(messagesRef);
        const messages = snapshot.val();
        let messageCount = messages ? Object.keys(messages).length : 0;
        
        if (messageCount > 0 && messageCount % MESSAGES_PER_EPOCH === 0) {
            currentEpoch = incrementEpoch(conversationId);
        }
        
        const encrypted = await encryptMessage(
            messageText,
            masterSecret,
            conversationId,
            currentEpoch
        );
        if (!encrypted) {
            throw new Error('Encryption failed');
        }
        
        const messagesRef2 = ref(db, `messages/${conversationId}`);
        await push(messagesRef2, {
            sender: senderUid,
            ciphertext: encrypted.ciphertext,
            nonce: encrypted.nonce,
            epoch: encrypted.epoch,
            timestamp: Date.now()
        });
        
        cleanupEpochKeys(conversationId, currentEpoch);
        
        console.log(`✅ Message sent with epoch ${currentEpoch} for ${conversationId}`);
        return true;
    } catch (error) {
        console.error('Send message error:', error);
        return false;
    }
};

// ============ LISTEN FOR MESSAGES ============
export const listenForMessages = (userUid, contactUid, callback) => {
    const conversationId = [userUid, contactUid].sort().join('_');
    const messagesRef = ref(db, `messages/${conversationId}`);
    const messagesQuery = query(messagesRef, limitToLast(50));
    
    const unsubscribe = onChildAdded(messagesQuery, async (snapshot) => {
        const messageData = snapshot.val();
        const messageId = snapshot.key;
        
        if (!messageData) return;
        
        try {
            const myKeys = getKeys();
            if (!myKeys) {
                console.error('No keys found');
                return;
            }
            
            const otherUid = messageData.sender === userUid ? contactUid : messageData.sender;
            const otherPublicKey = await getPublicKey(otherUid);
            
            if (!otherPublicKey) {
                console.error('Other user has no public key');
                return;
            }
            
            const masterSecret = await deriveSharedSecret(
                myKeys.privateKey,
                otherPublicKey
            );
            
            if (!masterSecret) {
                console.error('Failed to derive master secret');
                return;
            }
            
            const decrypted = await decryptMessage(
                messageData.ciphertext,
                messageData.nonce,
                masterSecret,
                conversationId,
                messageData.epoch || 1
            );
            
            if (decrypted) {
                const isSent = messageData.sender === userUid;
                callback({
                    ...messageData,
                    decryptedText: decrypted
                }, messageId, isSent);
            }
        } catch (error) {
            console.error('Decryption error:', error);
        }
    });
    
    return unsubscribe;
};

// ============ LOAD MESSAGES ============
export const loadMessages = async (userUid, contactUid) => {
    try {
        const conversationId = [userUid, contactUid].sort().join('_');
        const messagesRef = ref(db, `messages/${conversationId}`);
        const snapshot = await get(messagesRef);
        const data = snapshot.val();
        
        if (!data) return [];
        
        const myKeys = getKeys();
        if (!myKeys) {
            console.error('No keys found');
            return [];
        }
        
        // ✅ SIMPLIFIED: Just use contactUid directly
        const otherPublicKey = await getPublicKey(contactUid);
        if (!otherPublicKey) {
            console.error('Other user has no public key');
            return [];
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            otherPublicKey
        );
        
        const messages = [];
        for (const [id, msg] of Object.entries(data)) {
            if (!msg) continue;
            
            try {
                const decrypted = await decryptMessage(
                    msg.ciphertext,
                    msg.nonce,
                    masterSecret,
                    conversationId,
                    msg.epoch || 1
                );
                
                messages.push({
                    id: id,
                    ...msg,
                    isSent: msg.sender === userUid,
                    text: decrypted || 'Decryption failed'
                });
            } catch (error) {
                console.error('Error loading message:', error);
                messages.push({
                    id: id,
                    ...msg,
                    isSent: msg.sender === userUid,
                    text: 'Error loading'
                });
            }
        }
        
        return messages.sort((a, b) => a.timestamp - b.timestamp);
    } catch (error) {
        console.error('Load messages error:', error);
        return [];
    }
};

// ============ DELETE MESSAGE ============
export const deleteMessage = async (userUid, conversationId, messageId, deleteForEveryone = false) => {
    try {
        const messageRef = ref(db, `messages/${conversationId}/${messageId}`);
        
        if (deleteForEveryone) {
            await remove(messageRef);
        } else {
            await update(messageRef, {
                deleted: true,
                deletedBy: userUid
            });
        }
        
        return true;
    } catch (error) {
        console.error('Delete message error:', error);
        return false;
    }
};

// ============ EDIT MESSAGE ============
export const editMessage = async (userUid, conversationId, messageId, newText) => {
    try {
        const myKeys = getKeys();
        if (!myKeys) {
            throw new Error('Encryption keys not found');
        }
        
        const [uid1, uid2] = conversationId.split('_');
        const otherUid = uid1 === userUid ? uid2 : uid1;
        
        const otherPublicKey = await getPublicKey(otherUid);
        if (!otherPublicKey) {
            throw new Error('Other user has no public key');
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            otherPublicKey
        );
        
        if (!masterSecret) {
            throw new Error('Failed to derive master secret');
        }
        
        const currentEpoch = getCurrentEpoch(conversationId);
        
        const encrypted = await encryptMessage(
            newText,
            masterSecret,
            conversationId,
            currentEpoch
        );
        if (!encrypted) {
            throw new Error('Encryption failed');
        }
        
        const messageRef = ref(db, `messages/${conversationId}/${messageId}`);
        await update(messageRef, {
            ciphertext: encrypted.ciphertext,
            nonce: encrypted.nonce,
            edited: true,
            editedAt: Date.now(),
            epoch: encrypted.epoch
        });
        
        return true;
    } catch (error) {
        console.error('Edit message error:', error);
        return false;
    }
};

// ============ CLEAR CHAT ============
export const clearChat = async (conversationId) => {
    try {
        const messagesRef = ref(db, `messages/${conversationId}`);
        await remove(messagesRef);
        conversationEpochs.set(conversationId, 1);
        
        // ✅ Flush the cache immediately instead of using cleanup
        clearEpochKeys(conversationId);
        
        return true;
    } catch (error) {
        console.error('Clear chat error:', error);
        return false;
    }
};