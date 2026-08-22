// src/components/Contacts/ContactList.jsx
import ContactItem from './ContactItem';

export default function ContactList({ 
    contactList, 
    selectedContact, 
    setSelectedContact 
}) {
    if (contactList.length === 0) {
        return (
            <div className="contacts-list">
                <div className="empty-state">No contacts yet. Search for users above.</div>
            </div>
        );
    }

    return (
        <div className="contacts-list">
            {contactList.map((contact) => (
                <ContactItem
                    key={contact.uid}
                    contact={contact}
                    isActive={selectedContact?.uid === contact.uid}
                    onSelect={setSelectedContact}
                />
            ))}
        </div>
    );
}