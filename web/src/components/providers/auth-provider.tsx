'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

import { authApi, type User, type LoginData, type RegisterData } from '@/lib/api/auth';
import { tokenStorage } from '@/lib/auth/token-storage';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (data: LoginData) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  const isAuthenticated = !!user;

  // Initialize auth state on mount
  useEffect(() => {
    initializeAuth();
  }, []);

  const initializeAuth = async () => {
    try {
      const token = tokenStorage.getToken();
      
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Verify token and get user info
      const userData = await authApi.getCurrentUser();
      setUser(userData);
    } catch (error) {
      // Token is invalid, clear it
      tokenStorage.clearTokens();
      console.error('Auth initialization failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (data: LoginData) => {
    try {
      setIsLoading(true);
      
      const response = await authApi.login(data);
      
      // Store tokens
      tokenStorage.setTokens(response.access_token, response.refresh_token);
      
      // Get user data
      const userData = await authApi.getCurrentUser();
      setUser(userData);
      
      toast.success('Welcome back!');
      
      // Redirect to dashboard or intended page
      const redirectPath = new URLSearchParams(window.location.search).get('redirect') || '/dashboard';
      router.push(redirectPath);
      
    } catch (error: any) {
      const message = error?.response?.data?.detail || 'Login failed. Please try again.';
      toast.error(message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (data: RegisterData) => {
    try {
      setIsLoading(true);
      
      await authApi.register(data);
      
      toast.success('Account created successfully! Please log in.');
      
      // Redirect to login
      router.push('/auth/login');
      
    } catch (error: any) {
      const message = error?.response?.data?.detail || 'Registration failed. Please try again.';
      toast.error(message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    // Clear tokens and user state
    tokenStorage.clearTokens();
    setUser(null);
    
    toast.success('Logged out successfully');
    
    // Redirect to home page
    router.push('/');
  };

  const refreshUser = async () => {
    try {
      const userData = await authApi.getCurrentUser();
      setUser(userData);
    } catch (error) {
      console.error('Failed to refresh user data:', error);
      // Don't logout on refresh failure, token might still be valid
    }
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
}

// Hook for protecting routes
export function useRequireAuth() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      // Redirect to login with current path as redirect target
      const currentPath = window.location.pathname + window.location.search;
      router.push(`/auth/login?redirect=${encodeURIComponent(currentPath)}`);
    }
  }, [isAuthenticated, isLoading, router]);

  return { isAuthenticated, isLoading };
}