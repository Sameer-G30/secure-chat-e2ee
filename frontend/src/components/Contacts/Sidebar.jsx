// src/components/Contacts/Sidebar.jsx
import { IconGear } from '../../icons';
import SearchBar from './SearchBar';
import SearchResults from './SearchResults';
import ContactList from './ContactList';

export default function Sidebar({
    currentUser,
    openSettings,
    searchTerm,
    setSearchTerm,
    handleSearchUsers,
    isSearching,
    searchResults,
    handleAddContact,
    contactList,
    selectedContact,
    setSelectedContact,
    profilePicture
}) {
    return (
        <div className="sidebar">
            {/* Sidebar Header */}
            <div className="sidebar-header">
                <div className="user-avatar">
                    {profilePicture ? (
                        <img 
                            src={profilePicture} 
                            alt="Profile" 
                            style={{ 
                                width: '100%', 
                                height: '100%', 
                                borderRadius: '50%', 
                                objectFit: 'cover' 
                            }} 
                        />
                    ) : (
                        currentUser?.displayName?.charAt(0).toUpperCase() || 
                        currentUser?.email?.charAt(0).toUpperCase() || 
                        'U'
                    )}
                </div>
                <div className="user-info">
                    <div className="user-name">
                        {currentUser?.displayName || currentUser?.email?.split('@')[0] || 'User'}
                    </div>
                    <div className="user-status">Online</div>
                </div>
                <div className="sidebar-actions">
                    <button className="icon-btn" onClick={openSettings} title="Settings">
                        <IconGear />
                    </button>
                </div>
            </div>

            {/* Search */}
            <SearchBar
                searchTerm={searchTerm}
                setSearchTerm={setSearchTerm}
                handleSearchUsers={handleSearchUsers}
                isSearching={isSearching}
            />

            {/* Search Results */}
            <SearchResults
                searchResults={searchResults}
                handleAddContact={handleAddContact}
            />

            {/* Contacts List */}
            <ContactList
                contactList={contactList}
                selectedContact={selectedContact}
                setSelectedContact={setSelectedContact}
            />
        </div>
    );
}