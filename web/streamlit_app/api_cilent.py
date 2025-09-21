"""
Innomatics OMR System - API Client for Streamlit
Production-grade client for backend communication
"""

import requests
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional, Tuple
import streamlit as st
import logging
from datetime import datetime
import time

class InnoMaticsAPIClient:
    """Production API client for OMR backend services"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'InnoMaticsOMR/1.0'
        })
        self.token = None
        
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    def set_auth_token(self, token: str):
        """Set authentication token"""
        self.token = token
        self.session.headers.update({
            'Authorization': f'Bearer {token}'
        })
    
    def login(self, username: str, password: str) -> Dict:
        """Authenticate user and get token"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={'username': username, 'password': password},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.set_auth_token(data.get('access_token'))
                return {
                    'success': True,
                    'user': data.get('user', {}),
                    'token': data.get('access_token')
                }
            else:
                return {
                    'success': False,
                    'error': f"Authentication failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }
    
    def health_check(self) -> Dict:
        """Check API health status"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Health check failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"API unavailable: {str(e)}"
            }
    
    def upload_and_process(self, files: List, config: Dict) -> Dict:
        """Upload files and start processing"""
        try:
            # Prepare files for upload
            files_data = []
            for file in files:
                files_data.append(
                    ('files', (file.name, file.getvalue(), file.type))
                )
            
            # Prepare form data
            form_data = {
                'sheet_version': config.get('sheet_version', 'SET-A'),
                'confidence_threshold': config.get('confidence_threshold', 0.8),
                'enable_ml': config.get('enable_ml', False),
                'concurrent': config.get('concurrent', True)
            }
            
            # Make request
            response = self.session.post(
                f"{self.base_url}/api/v1/omr/process-batch",
                files=files_data,
                data=form_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Upload failed: {response.status_code} - {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Upload error: {str(e)}"
            }
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get processing job status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/jobs/{job_id}/status",
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Status check failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Status check error: {str(e)}"
            }
    
    def get_job_results(self, job_id: str) -> Dict:
        """Get processing job results"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/jobs/{job_id}/results",
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Results fetch failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Results fetch error: {str(e)}"
            }
    
    def get_answer_keys(self) -> Dict:
        """Get available answer key versions"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/answer-keys",
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Answer keys fetch failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Answer keys error: {str(e)}"
            }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/stats",
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"Stats fetch failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Stats error: {str(e)}"
            }

# Streamlit integration functions
class StreamlitAPIIntegration:
    """Streamlit-specific API integration helpers"""
    
    def __init__(self, api_client: InnoMaticsAPIClient):
        self.api = api_client
    
    def display_connection_status(self):
        """Display API connection status in Streamlit"""
        health = self.api.health_check()
        
        if health['success']:
            st.success("🟢 API Connected")
            
            with st.expander("API Status Details"):
                data = health['data']
                st.json(data)
        else:
            st.error(f"🔴 API Disconnected: {health['error']}")
    
    def process_files_with_progress(self, files: List, config: Dict) -> Optional[Dict]:
        """Process files with real-time progress display"""
        
        if not files:
            st.error("No files to process")
            return None
        
        # Start processing
        with st.spinner("🚀 Starting processing..."):
            upload_result = self.api.upload_and_process(files, config)
        
        if not upload_result['success']:
            st.error(f"Upload failed: {upload_result['error']}")
            return None
        
        job_id = upload_result['data']['job_id']
        st.success(f"✅ Processing started! Job ID: {job_id}")
        
        # Progress tracking
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        details_expander = st.expander("Processing Details", expanded=False)
        
        while True:
            status_result = self.api.get_job_status(job_id)
            
            if not status_result['success']:
                st.error(f"Status check failed: {status_result['error']}")
                break
            
            job_data = status_result['data']
            progress = job_data['progress'] / 100.0
            status = job_data['status']
            
            # Update progress
            progress_bar.progress(progress)
            status_text.text(f"Status: {status.title()} - {job_data['processed_files']}/{job_data['total_files']} files")
            
            # Show details
            with details_expander:
                st.json(job_data)
            
            # Check if completed
            if status in ['completed', 'failed']:
                break
            
            # Wait before next update
            time.sleep(2)
        
        # Get final results
        if status == 'completed':
            st.success("🎉 Processing completed successfully!")
            
            results_response = self.api.get_job_results(job_id)
            if results_response['success']:
                return results_response['data']
            else:
                st.error(f"Failed to fetch results: {results_response['error']}")
                return None
        else:
            st.error("❌ Processing failed")
            return None
    
    def display_results_dashboard(self, results_data: Dict):
        """Display processing results in dashboard format"""
        if not results_data:
            st.info("No results to display")
            return
        
        results = results_data.get('results', [])
        summary = results_data.get('summary', {})
        
        # Summary metrics
        st.subheader("📊 Processing Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", summary.get('total_files', 0))
        
        with col2:
            st.metric("Successful", summary.get('successful', 0))
        
        with col3:
            st.metric("Average Score", f"{summary.get('average_score', 0):.1f}%")
        
        with col4:
            st.metric("Average Confidence", f"{summary.get('average_confidence', 0):.3f}")
        
        # Results table
        if results:
            st.subheader("📋 Detailed Results")
            
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Format for display
            display_df = df[['sheet_id', 'success', 'percentage_score', 'overall_confidence', 'status']]
            display_df.columns = ['Sheet ID', 'Success', 'Score %', 'Confidence', 'Status']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    'Success': st.column_config.CheckboxColumn(),
                    'Score %': st.column_config.ProgressColumn('Score %', min_value=0, max_value=100),
                    'Confidence': st.column_config.ProgressColumn('Confidence', min_value=0.0, max_value=1.0)
                }
            )
            
            # Export option
            if st.button("📥 Export Results"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Usage example for Streamlit apps
def get_api_client() -> InnoMaticsAPIClient:
    """Get API client instance (cached in Streamlit session)"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = InnoMaticsAPIClient()
    
    return st.session_state.api_client

def get_streamlit_integration() -> StreamlitAPIIntegration:
    """Get Streamlit integration helper (cached)"""
    if 'api_integration' not in st.session_state:
        api_client = get_api_client()
        st.session_state.api_integration = StreamlitAPIIntegration(api_client)
    
    return st.session_state.api_integration
