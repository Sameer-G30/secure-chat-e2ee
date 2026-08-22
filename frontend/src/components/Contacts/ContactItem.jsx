// src/components/Contacts/ContactItem.jsx
export default function ContactItem({ contact, isActive, onSelect }) {
    return (
        <div
            className={`contact-item ${isActive ? 'active' : ''}`}
            onClick={() => onSelect(contact)}
        >
            <div className="contact-avatar">
                {contact.username.charAt(0).toUpperCase()}
            </div>
            <div className="contact-info">
                <div className="contact-name">{contact.username}</div>
                <div className="contact-preview">Tap to chat</div>
            </div>
        </div>
    );
}