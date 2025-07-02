"""
API Key Management Agent - Secure Credential Management
=====================================================

This agent securely manages API keys, usernames, passwords, and other
credentials through encryption, rotation, and access control.

CRITICAL SYSTEM POLICY: NO TRADING DECISIONS OR RECOMMENDATIONS
This agent only performs credential management, security operations, and
access control. No trading advice is provided.
"""

import asyncio
import logging
import json
import os
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Credential:
    """Represents a managed credential"""
    name: str
    type: str  # 'api_key', 'username', 'password', 'token', 'certificate'
    value: str
    encrypted_value: str
    provider: str
    description: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class SecurityPolicy:
    """Security policy for credential management"""
    min_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    require_lowercase: bool = True
    max_age_days: int = 90
    rotation_interval_days: int = 30
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 15

class APIKeyManagementAgent:
    """
    Secure API key and credential management agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "APIKeyManagementAgent"
        self.version = "1.0.0"
        self.master_key = None
        self.cipher_suite = None
        self.credentials = {}
        self.security_policy = SecurityPolicy()
        self.failed_attempts = {}
        self.locked_accounts = {}
        
    async def initialize(self):
        """Initialize the API key management agent"""
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # AI REASONING: Initialize secure credential management
        # PSEUDOCODE:
        # 1. Generate or load master encryption key
        # 2. Initialize cryptographic cipher suite
        # 3. Set up secure storage and access controls
        # 4. Configure security policies and validation rules
        # 5. Initialize credential tracking and monitoring
        # 6. Set up audit logging and compliance tracking
        # 7. Configure automatic rotation and expiration
        # 8. Initialize MCP communication for security events
        
        try:
            # Initialize encryption
            await self._initialize_encryption()
            
            # Load existing credentials
            await self._load_credentials()
            
            # Set up security monitoring
            await self._setup_security_monitoring()
            
        except Exception as e:
            logger.error(f"Error initializing API key management: {e}")
            raise
            
        logger.info(f"{self.name} initialized successfully")
    
    async def _initialize_encryption(self):
        """
        Initialize encryption capabilities
        """
        # AI REASONING: Encryption initialization
        # PSEUDOCODE:
        # 1. Generate secure master key using cryptographically secure RNG
        # 2. Derive encryption key using PBKDF2 with high iteration count
        # 3. Initialize Fernet cipher suite for symmetric encryption
        # 4. Validate encryption setup and key strength
        # 5. Set up key rotation and backup mechanisms
        # 6. Configure secure key storage and access controls
        
        try:
            # Try to load existing master key
            master_key = keyring.get_password("api_key_manager", "master_key")
            
            if not master_key:
                # Generate new master key
                master_key = Fernet.generate_key()
                keyring.set_password("api_key_manager", "master_key", master_key.decode())
            
            self.master_key = master_key if isinstance(master_key, bytes) else master_key.encode()
            self.cipher_suite = Fernet(self.master_key)
            
            logger.info("Encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            raise
    
    async def _load_credentials(self):
        """
        Load existing credentials from secure storage
        """
        # AI REASONING: Credential loading and validation
        # PSEUDOCODE:
        # 1. Load encrypted credentials from secure storage
        # 2. Decrypt and validate credential integrity
        # 3. Check for expired or compromised credentials
        # 4. Update usage statistics and access logs
        # 5. Validate credential format and structure
        # 6. Generate credential health report
        
        try:
            # Load from secure storage (in production, use proper database)
            credentials_file = "credentials.json.enc"
            if os.path.exists(credentials_file):
                with open(credentials_file, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                    credentials_data = json.loads(decrypted_data.decode())
                    
                    for cred_data in credentials_data:
                        credential = Credential(**cred_data)
                        self.credentials[credential.name] = credential
                        
                logger.info(f"Loaded {len(self.credentials)} credentials")
                
        except Exception as e:
            logger.warning(f"Error loading credentials: {e}")
    
    async def _setup_security_monitoring(self):
        """
        Set up security monitoring and alerts
        """
        # AI REASONING: Security monitoring setup
        # PSEUDOCODE:
        # 1. Initialize access monitoring and logging
        # 2. Set up failed attempt tracking and lockout mechanisms
        # 3. Configure security event alerts and notifications
        # 4. Initialize credential health monitoring
        # 5. Set up compliance and audit tracking
        # 6. Configure automatic security assessments
        
        logger.info("Security monitoring initialized")
    
    async def store_credential(self, name: str, credential_type: str, value: str, 
                             provider: str, description: str = "", 
                             expires_at: Optional[datetime] = None,
                             tags: List[str] = None) -> bool:
        """
        Store a new credential securely
        """
        # AI REASONING: Secure credential storage
        # PSEUDOCODE:
        # 1. Validate credential format and strength
        # 2. Check for duplicate credentials and conflicts
        # 3. Encrypt credential value using strong encryption
        # 4. Generate secure credential metadata
        # 5. Store encrypted credential in secure storage
        # 6. Update credential registry and tracking
        # 7. Generate audit log entry
        # 8. Validate storage success and integrity
        
        try:
            # Validate credential
            if not await self._validate_credential(credential_type, value):
                logger.error(f"Invalid credential format for {name}")
                return False
            
            # Check for duplicates
            if name in self.credentials:
                logger.warning(f"Credential {name} already exists")
                return False
            
            # Encrypt the value
            encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
            
            # Create credential object
            credential = Credential(
                name=name,
                type=credential_type,
                value=value,
                encrypted_value=encrypted_value,
                provider=provider,
                description=description,
                created_at=datetime.now(),
                expires_at=expires_at,
                tags=tags or []
            )
            
            # Store credential
            self.credentials[name] = credential
            
            # Save to secure storage
            await self._save_credentials()
            
            logger.info(f"Stored credential: {name} ({credential_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error storing credential {name}: {e}")
            return False
    
    async def _validate_credential(self, credential_type: str, value: str) -> bool:
        """
        Validate credential format and strength
        """
        # AI REASONING: Credential validation
        # PSEUDOCODE:
        # 1. Check credential type-specific format requirements
        # 2. Validate length and complexity requirements
        # 3. Check for common weak patterns and vulnerabilities
        # 4. Assess entropy and randomness quality
        # 5. Validate against security policy requirements
        # 6. Generate validation score and recommendations
        
        if not value or len(value.strip()) == 0:
            return False
        
        if credential_type == "password":
            return self._validate_password_strength(value)
        elif credential_type == "api_key":
            return self._validate_api_key_format(value)
        elif credential_type == "username":
            return self._validate_username_format(value)
        elif credential_type == "token":
            return self._validate_token_format(value)
        
        return True
    
    def _validate_password_strength(self, password: str) -> bool:
        """
        Validate password strength
        """
        # AI REASONING: Password strength assessment
        # PSEUDOCODE:
        # 1. Check minimum length requirements
        # 2. Validate character set diversity
        # 3. Check for common weak patterns
        # 4. Assess entropy and randomness
        # 5. Generate strength score and recommendations
        
        if len(password) < self.security_policy.min_length:
            return False
        
        if self.security_policy.require_uppercase and not re.search(r'[A-Z]', password):
            return False
        
        if self.security_policy.require_lowercase and not re.search(r'[a-z]', password):
            return False
        
        if self.security_policy.require_numbers and not re.search(r'\d', password):
            return False
        
        if self.security_policy.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        
        return True
    
    def _validate_api_key_format(self, api_key: str) -> bool:
        """
        Validate API key format
        """
        # AI REASONING: API key format validation
        # PSEUDOCODE:
        # 1. Check for appropriate length and format
        # 2. Validate character set and encoding
        # 3. Check for common API key patterns
        # 4. Assess key strength and uniqueness
        
        if len(api_key) < 16:
            return False
        
        # Check for common API key patterns
        if re.match(r'^[A-Za-z0-9_-]+$', api_key):
            return True
        
        return False
    
    def _validate_username_format(self, username: str) -> bool:
        """
        Validate username format
        """
        # AI REASONING: Username format validation
        # PSEUDOCODE:
        # 1. Check for appropriate length and characters
        # 2. Validate against common username patterns
        # 3. Check for reserved or problematic usernames
        
        if len(username) < 3 or len(username) > 50:
            return False
        
        if re.match(r'^[A-Za-z0-9_-]+$', username):
            return True
        
        return False
    
    def _validate_token_format(self, token: str) -> bool:
        """
        Validate token format
        """
        # AI REASONING: Token format validation
        # PSEUDOCODE:
        # 1. Check for appropriate length and format
        # 2. Validate character set and encoding
        # 3. Check for common token patterns (JWT, etc.)
        
        if len(token) < 20:
            return False
        
        return True
    
    async def retrieve_credential(self, name: str, access_key: str = None) -> Optional[str]:
        """
        Retrieve a credential value
        """
        # AI REASONING: Secure credential retrieval
        # PSEUDOCODE:
        # 1. Validate access permissions and authentication
        # 2. Check for account lockouts and security restrictions
        # 3. Decrypt credential value securely
        # 4. Update usage statistics and access logs
        # 5. Check for credential expiration and health
        # 6. Generate audit trail entry
        # 7. Validate retrieval success and integrity
        
        try:
            # Check if credential exists
            if name not in self.credentials:
                logger.warning(f"Credential {name} not found")
                return None
            
            credential = self.credentials[name]
            
            # Check if credential is active
            if not credential.is_active:
                logger.warning(f"Credential {name} is inactive")
                return None
            
            # Check for expiration
            if credential.expires_at and credential.expires_at < datetime.now():
                logger.warning(f"Credential {name} has expired")
                return None
            
            # Decrypt the value
            decrypted_value = self.cipher_suite.decrypt(
                credential.encrypted_value.encode()
            ).decode()
            
            # Update usage statistics
            credential.last_used = datetime.now()
            credential.usage_count += 1
            
            # Save updated statistics
            await self._save_credentials()
            
            logger.info(f"Retrieved credential: {name}")
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Error retrieving credential {name}: {e}")
            return None
    
    async def rotate_credential(self, name: str, new_value: str = None) -> bool:
        """
        Rotate a credential (generate new value or update existing)
        """
        # AI REASONING: Credential rotation and renewal
        # PSEUDOCODE:
        # 1. Validate current credential and access permissions
        # 2. Generate new secure credential value
        # 3. Update credential in secure storage
        # 4. Maintain credential history and audit trail
        # 5. Update expiration and rotation timestamps
        # 6. Notify dependent systems of credential change
        # 7. Validate rotation success and integrity
        
        try:
            if name not in self.credentials:
                logger.error(f"Credential {name} not found for rotation")
                return False
            
            credential = self.credentials[name]
            
            # Generate new value if not provided
            if not new_value:
                new_value = await self._generate_credential_value(credential.type)
            
            # Validate new value
            if not await self._validate_credential(credential.type, new_value):
                logger.error(f"Invalid new credential value for {name}")
                return False
            
            # Encrypt new value
            encrypted_value = self.cipher_suite.encrypt(new_value.encode()).decode()
            
            # Update credential
            credential.value = new_value
            credential.encrypted_value = encrypted_value
            credential.created_at = datetime.now()
            credential.expires_at = datetime.now() + timedelta(days=self.security_policy.max_age_days)
            
            # Save updated credential
            await self._save_credentials()
            
            logger.info(f"Rotated credential: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating credential {name}: {e}")
            return False
    
    async def _generate_credential_value(self, credential_type: str) -> str:
        """
        Generate a secure credential value
        """
        # AI REASONING: Secure credential generation
        # PSEUDOCODE:
        # 1. Determine appropriate generation algorithm based on type
        # 2. Use cryptographically secure random number generation
        # 3. Apply type-specific format and length requirements
        # 4. Validate generated value against security policies
        # 5. Ensure uniqueness and avoid collisions
        # 6. Generate entropy assessment and quality metrics
        
        if credential_type == "password":
            return self._generate_secure_password()
        elif credential_type == "api_key":
            return self._generate_api_key()
        elif credential_type == "token":
            return self._generate_token()
        else:
            return secrets.token_urlsafe(32)
    
    def _generate_secure_password(self) -> str:
        """
        Generate a secure password
        """
        # AI REASONING: Secure password generation
        # PSEUDOCODE:
        # 1. Use cryptographically secure random generation
        # 2. Ensure character set diversity and complexity
        # 3. Apply length and complexity requirements
        # 4. Validate against common weak patterns
        # 5. Generate entropy assessment
        
        import string
        
        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*(),.?\":{}|<>"
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill remaining length with random characters
        all_chars = lowercase + uppercase + digits + special
        remaining_length = self.security_policy.min_length - 4
        
        for _ in range(remaining_length):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        password_list = list(password)
        secrets.SystemRandom().shuffle(password_list)
        
        return ''.join(password_list)
    
    def _generate_api_key(self) -> str:
        """
        Generate a secure API key
        """
        # AI REASONING: API key generation
        # PSEUDOCODE:
        # 1. Use cryptographically secure random generation
        # 2. Apply API key format and length requirements
        # 3. Ensure uniqueness and avoid collisions
        # 4. Generate entropy assessment
        
        return secrets.token_urlsafe(32)
    
    def _generate_token(self) -> str:
        """
        Generate a secure token
        """
        # AI REASONING: Token generation
        # PSEUDOCODE:
        # 1. Use cryptographically secure random generation
        # 2. Apply token format and length requirements
        # 3. Ensure uniqueness and avoid collisions
        # 4. Generate entropy assessment
        
        return secrets.token_urlsafe(48)
    
    async def revoke_credential(self, name: str) -> bool:
        """
        Revoke a credential (mark as inactive)
        """
        # AI REASONING: Credential revocation
        # PSEUDOCODE:
        # 1. Validate credential existence and access permissions
        # 2. Mark credential as inactive
        # 3. Update revocation timestamp and reason
        # 4. Generate audit trail entry
        # 5. Notify dependent systems of revocation
        # 6. Update credential registry and tracking
        
        try:
            if name not in self.credentials:
                logger.error(f"Credential {name} not found for revocation")
                return False
            
            credential = self.credentials[name]
            credential.is_active = False
            
            # Save updated credential
            await self._save_credentials()
            
            logger.info(f"Revoked credential: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking credential {name}: {e}")
            return False
    
    async def list_credentials(self, filter_type: str = None, 
                             include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        List credentials with optional filtering
        """
        # AI REASONING: Credential listing and filtering
        # PSEUDOCODE:
        # 1. Validate access permissions and authentication
        # 2. Apply filtering criteria and access controls
        # 3. Generate sanitized credential information
        # 4. Exclude sensitive data from listing
        # 5. Apply pagination and result limits
        # 6. Generate audit trail entry
        
        credentials_list = []
        
        for name, credential in self.credentials.items():
            if not include_inactive and not credential.is_active:
                continue
            
            if filter_type and credential.type != filter_type:
                continue
            
            # Create sanitized credential info
            cred_info = {
                "name": credential.name,
                "type": credential.type,
                "provider": credential.provider,
                "description": credential.description,
                "created_at": credential.created_at.isoformat(),
                "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
                "last_used": credential.last_used.isoformat() if credential.last_used else None,
                "usage_count": credential.usage_count,
                "is_active": credential.is_active,
                "tags": credential.tags
            }
            
            credentials_list.append(cred_info)
        
        return credentials_list
    
    async def _save_credentials(self):
        """
        Save credentials to secure storage
        """
        # AI REASONING: Secure storage operations
        # PSEUDOCODE:
        # 1. Serialize credential data securely
        # 2. Encrypt all sensitive data
        # 3. Validate data integrity before saving
        # 4. Use secure storage mechanisms
        # 5. Generate backup and recovery options
        # 6. Validate save operation success
        
        try:
            # Convert credentials to serializable format
            credentials_data = []
            for credential in self.credentials.values():
                cred_dict = {
                    "name": credential.name,
                    "type": credential.type,
                    "value": credential.value,
                    "encrypted_value": credential.encrypted_value,
                    "provider": credential.provider,
                    "description": credential.description,
                    "created_at": credential.created_at.isoformat(),
                    "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
                    "last_used": credential.last_used.isoformat() if credential.last_used else None,
                    "usage_count": credential.usage_count,
                    "is_active": credential.is_active,
                    "tags": credential.tags
                }
                credentials_data.append(cred_dict)
            
            # Encrypt and save
            data_json = json.dumps(credentials_data)
            encrypted_data = self.cipher_suite.encrypt(data_json.encode())
            
            with open("credentials.json.enc", 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            raise
    
    async def run_security_audit(self) -> Dict[str, Any]:
        """
        Run security audit on all credentials
        """
        # AI REASONING: Security audit and assessment
        # PSEUDOCODE:
        # 1. Analyze all credentials for security issues
        # 2. Check for expired or weak credentials
        # 3. Identify potential security vulnerabilities
        # 4. Assess credential usage patterns and anomalies
        # 5. Generate security recommendations
        # 6. Create compliance and audit reports
        # 7. Identify credential rotation needs
        # 8. Generate security score and metrics
        
        logger.info("Starting security audit")
        
        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "total_credentials": len(self.credentials),
            "active_credentials": len([c for c in self.credentials.values() if c.is_active]),
            "expired_credentials": [],
            "weak_credentials": [],
            "unused_credentials": [],
            "security_score": 0.0,
            "recommendations": []
        }
        
        try:
            for name, credential in self.credentials.items():
                # Check for expired credentials
                if credential.expires_at and credential.expires_at < datetime.now():
                    audit_results["expired_credentials"].append(name)
                
                # Check for weak credentials
                if not await self._validate_credential(credential.type, credential.value):
                    audit_results["weak_credentials"].append(name)
                
                # Check for unused credentials
                if credential.usage_count == 0 and credential.created_at < datetime.now() - timedelta(days=30):
                    audit_results["unused_credentials"].append(name)
            
            # Calculate security score
            total_issues = (len(audit_results["expired_credentials"]) + 
                          len(audit_results["weak_credentials"]) + 
                          len(audit_results["unused_credentials"]))
            
            audit_results["security_score"] = max(0.0, 1.0 - (total_issues / len(self.credentials)))
            
            # Generate recommendations
            if audit_results["expired_credentials"]:
                audit_results["recommendations"].append("Rotate expired credentials")
            
            if audit_results["weak_credentials"]:
                audit_results["recommendations"].append("Strengthen weak credentials")
            
            if audit_results["unused_credentials"]:
                audit_results["recommendations"].append("Review and potentially revoke unused credentials")
            
            logger.info(f"Security audit completed. Score: {audit_results['security_score']:.2f}")
            return audit_results
            
        except Exception as e:
            logger.error(f"Error in security audit: {e}")
            audit_results["recommendations"].append(f"Audit error: {e}")
            return audit_results
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} cleanup completed")

# Example usage
async def main():
    config = {
        "encryption_key_file": "master.key",
        "credentials_file": "credentials.json.enc",
        "backup_enabled": True
    }
    
    agent = APIKeyManagementAgent(config)
    await agent.initialize()
    
    try:
        # Store a test credential
        success = await agent.store_credential(
            name="test_api_key",
            credential_type="api_key",
            value="test_value_12345",
            provider="test_provider",
            description="Test API key for demonstration"
        )
        
        if success:
            # Retrieve the credential
            value = await agent.retrieve_credential("test_api_key")
            print(f"Retrieved value: {value}")
            
            # List credentials
            credentials = await agent.list_credentials()
            print(f"Total credentials: {len(credentials)}")
            
            # Run security audit
            audit = await agent.run_security_audit()
            print(f"Security score: {audit['security_score']:.2f}")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 