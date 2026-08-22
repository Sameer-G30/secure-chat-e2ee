// src/hooks/useContacts.js
import { useState, useEffect } from 'react';
import { searchUsers, addContact, getContacts, listenForContacts } from '../utils/contacts';

export function useContacts(currentUser) {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [contactList, setContactList] = useState([]);
    const [selectedContact, setSelectedContact] = useState(null);

    const loadContacts = async () => {
        if (!currentUser) return;
        
        try {
            const contacts = await getContacts(currentUser.uid);
            setContactList(contacts);
        } catch (error) {
            console.error('Load contacts error:', error);
        }
    };

    const handleSearchUsers = async () => {
        if (!searchTerm.trim() || searchTerm.length < 2) {
            setSearchResults([]);
            return;
        }
        
        setIsSearching(true);
        
        try {
            const results = await searchUsers(searchTerm);
            const filtered = results.filter(user => user.uid !== currentUser.uid);
            setSearchResults(filtered);
        } catch (error) {
            console.error('Search error:', error);
        }
        
        setIsSearching(false);
    };

    const handleAddContact = async (contactUid, contactUsername) => {
        if (!currentUser) return;
        
        try {
            const success = await addContact(currentUser.uid, contactUid, contactUsername);
            if (success) {
                alert(`Added ${contactUsername} to contacts!`);
                setSearchResults([]);
                setSearchTerm('');
                loadContacts();
            }
        } catch (error) {
            console.error('Add contact error:', error);
        }
    };

    // Listen for new contacts
    useEffect(() => {
        if (!currentUser) return;
        
        loadContacts();
        
        const unsubscribe = listenForContacts(currentUser.uid, (newContact) => {
            setContactList(prev => [...prev, newContact]);
        });
        
        return () => {
            if (unsubscribe) unsubscribe();
        };
    }, [currentUser]);

    return {
        searchTerm,
        setSearchTerm,
        searchResults,
        isSearching,
        contactList,
        selectedContact,
        setSelectedContact,
        handleSearchUsers,
        handleAddContact,
        loadContacts
    };
}