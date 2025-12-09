"""
Production-Ready AI-Powered Deepfake Detection System
Extracts metadata and uses OpenAI API for intelligent analysis
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from dotenv import load_dotenv

# OpenAI API
try:
    from openai import OpenAI
except ImportError:
    print("Installing OpenAI package...")
    os.system("pip install openai")
    from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageVerdict(Enum):
    """Image classification"""
    AI_GENERATED = "ai_generated"
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"


class RiskLevel(Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


@dataclass
class AnalysisResult:
    """Structured analysis result"""
    file_path: str
    file_hash: str
    verdict: str
    confidence_score: float  # 0-100
    risk_level: str
    reasoning: str
    key_findings: list
    ai_model_detected: Optional[str]
    camera_detected: Optional[str]
    metadata_summary: dict
    full_metadata: dict
    timestamp: str
    analysis_time_ms: float
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class MetadataExtractor:
    """Extract comprehensive metadata from images"""
    
    def parse_xmp_metadata(self, xmp_data) -> Dict:
        """Parse XMP metadata from bytes or string"""
        try:
            if isinstance(xmp_data, bytes):
                xmp_data = xmp_data.decode('utf-8', errors='ignore')
            
            root = ET.fromstring(xmp_data)
            namespaces = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'exif': 'http://ns.adobe.com/exif/1.0/',
                'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
                'Iptc4xmpExt': 'http://iptc.org/std/Iptc4xmpExt/2008-02-29/',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }
            
            xmp_dict = {}
            for desc in root.findall('.//rdf:Description', namespaces):
                for key, value in desc.attrib.items():
                    clean_key = key.split('}')[-1] if '}' in key else key
                    xmp_dict[clean_key] = value
            
            return xmp_dict
        except Exception as e:
            logger.warning(f"XMP parsing error: {str(e)}")
            return {}
    
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract all available metadata from image"""
        try:
            img = Image.open(image_path)
            metadata = {
                'basic_info': {
                    'filename': os.path.basename(image_path),
                    'format': img.format,
                    'mode': img.mode,
                    'size': list(img.size),
                    'width': img.width,
                    'height': img.height
                },
                'exif_data': {},
                'gps_data': {},
                'xmp_data': {}
            }
            
            # Process PIL info
            for key, value in img.info.items():
                if key in ['xmp', 'XML:com.adobe.xmp']:
                    metadata['xmp_data'] = self.parse_xmp_metadata(value)
                elif isinstance(value, bytes):
                    try:
                        metadata['basic_info'][key] = value.decode('utf-8', errors='ignore')
                    except:
                        metadata['basic_info'][key] = f"<binary: {len(value)} bytes>"
                else:
                    metadata['basic_info'][key] = str(value)
            
            # Extract EXIF
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    if tag == 'GPSInfo':
                        gps_info = {}
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_info[gps_tag] = str(gps_value)
                        metadata['gps_data'] = gps_info
                    else:
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = f"<binary: {len(value)} bytes>"
                        metadata['exif_data'][tag] = str(value)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
            return {'error': str(e)}


class OpenAIDeepfakeAnalyzer:
    """AI-powered deepfake detection using OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found! Set OPENAI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI client initialized successfully")
    
    def create_analysis_prompt(self, metadata: Dict) -> str:
        """Create detailed prompt for OpenAI analysis"""
        
        prompt = f"""You are an expert forensic analyst specializing in detecting AI-generated images and deepfakes. 

Analyze the following image metadata and determine if this image is:
1. AI-GENERATED (created by AI tools like Stable Diffusion, Midjourney, DALL-E, Gemini, Sora, etc.)
2. AUTHENTIC (captured by a real camera)
3. SUSPICIOUS (edited or unclear origin)
4. INCONCLUSIVE (insufficient data)

IMAGE METADATA:
================

BASIC INFO:
{json.dumps(metadata.get('basic_info', {}), indent=2)}

EXIF DATA:
{json.dumps(metadata.get('exif_data', {}), indent=2)}

XMP DATA:
{json.dumps(metadata.get('xmp_data', {}), indent=2)}

GPS DATA:
{json.dumps(metadata.get('gps_data', {}), indent=2)}

ANALYSIS INSTRUCTIONS:
======================

Look for these AI GENERATION INDICATORS:
- Credit field containing: "AI", "Generated", "Midjourney", "Stable Diffusion", "DALL-E", "Gemini", "Sora", "Firefly"
- DigitalSourceType: "trainedAlgorithmicMedia" or "algorithmicallyEnhanced"
- Software field with AI tool names
- Missing camera manufacturer (Make/Model)
- No EXIF data at all
- Timestamp too recent with no camera data
- Unusual or missing EXIF structure

Look for these AUTHENTICITY INDICATORS:
- Valid camera manufacturer (Make: Canon, Nikon, Sony, Apple, Samsung, Lenovo, etc.)
- Camera model present
- Camera software (e.g., "iPhone Camera", "Samsung Camera")
- Complete EXIF structure with camera settings
- GPS data (though absence doesn't mean fake)
- Timestamp from before 2022 (before AI image boom)

REQUIRED OUTPUT FORMAT (JSON only, no markdown):
{{
    "verdict": "ai_generated" | "authentic" | "suspicious" | "inconclusive",
    "confidence_score": 0-100,
    "reasoning": "Detailed explanation of your analysis",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "ai_model_detected": "model name or null",
    "camera_detected": "camera name or null",
    "risk_level": "critical" | "high" | "medium" | "low" | "safe"
}}

Provide ONLY the JSON response, no additional text."""

        return prompt
    
    def analyze_metadata(self, metadata: Dict) -> Dict:
        """Send metadata to OpenAI for analysis"""
        try:
            prompt = self.create_analysis_prompt(metadata)
            
            logger.info("Sending metadata to OpenAI for analysis...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 for best accuracy
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert forensic image analyst specializing in deepfake detection. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            result = response.choices[0].message.content
            analysis = json.loads(result)
            
            logger.info(f"OpenAI analysis complete: {analysis.get('verdict')}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {str(e)}")
            return {
                "verdict": "inconclusive",
                "confidence_score": 0,
                "reasoning": f"Analysis failed: Invalid response format",
                "key_findings": ["Error parsing AI response"],
                "ai_model_detected": None,
                "camera_detected": None,
                "risk_level": "medium"
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "verdict": "inconclusive",
                "confidence_score": 0,
                "reasoning": f"Analysis failed: {str(e)}",
                "key_findings": ["API error occurred"],
                "ai_model_detected": None,
                "camera_detected": None,
                "risk_level": "medium"
            }


class DeepfakeDetectionSystem:
    """Main production system combining metadata extraction and AI analysis"""
    
    def __init__(self, openai_api_key: Optional[str] = None, output_dir: str = "analysis_reports"):
        """Initialize the detection system"""
        self.metadata_extractor = MetadataExtractor()
        self.ai_analyzer = OpenAIDeepfakeAnalyzer(api_key=openai_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Deepfake Detection System initialized")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash for file tracking"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def analyze_image(self, image_path: str) -> AnalysisResult:
        """Complete analysis pipeline"""
        start_time = datetime.now()
        
        logger.info(f"Starting analysis: {image_path}")
        
        # Validate file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(image_path)
        
        # Extract metadata
        logger.info("Extracting metadata...")
        metadata = self.metadata_extractor.extract_metadata(image_path)
        
        if 'error' in metadata:
            logger.error(f"Metadata extraction failed: {metadata['error']}")
            return AnalysisResult(
                file_path=image_path,
                file_hash=file_hash,
                verdict=ImageVerdict.INCONCLUSIVE.value,
                confidence_score=0.0,
                risk_level=RiskLevel.MEDIUM.value,
                reasoning="Failed to extract metadata",
                key_findings=[metadata['error']],
                ai_model_detected=None,
                camera_detected=None,
                metadata_summary={},
                full_metadata=metadata,
                timestamp=datetime.now().isoformat(),
                analysis_time_ms=0
            )
        
        # Analyze with OpenAI
        logger.info("Analyzing with AI...")
        ai_analysis = self.ai_analyzer.analyze_metadata(metadata)
        
        # Create metadata summary
        metadata_summary = {
            'has_exif': bool(metadata.get('exif_data')),
            'has_xmp': bool(metadata.get('xmp_data')),
            'has_gps': bool(metadata.get('gps_data')),
            'camera_make': metadata.get('exif_data', {}).get('Make'),
            'camera_model': metadata.get('exif_data', {}).get('Model'),
            'software': metadata.get('exif_data', {}).get('Software'),
            'date_time': metadata.get('exif_data', {}).get('DateTime')
        }
        
        # Calculate analysis time
        end_time = datetime.now()
        analysis_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create result
        result = AnalysisResult(
            file_path=image_path,
            file_hash=file_hash,
            verdict=ai_analysis.get('verdict', 'inconclusive'),
            confidence_score=float(ai_analysis.get('confidence_score', 0)),
            risk_level=ai_analysis.get('risk_level', 'medium'),
            reasoning=ai_analysis.get('reasoning', 'No reasoning provided'),
            key_findings=ai_analysis.get('key_findings', []),
            ai_model_detected=ai_analysis.get('ai_model_detected'),
            camera_detected=ai_analysis.get('camera_detected'),
            metadata_summary=metadata_summary,
            full_metadata=metadata,
            timestamp=datetime.now().isoformat(),
            analysis_time_ms=round(analysis_time_ms, 2)
        )
        
        logger.info(f"Analysis complete: {result.verdict} ({result.confidence_score}%)")
        
        return result
    
    def save_report(self, result: AnalysisResult) -> Path:
        """Save detailed analysis report"""
        filename = f"analysis_{result.file_hash[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result.to_json())
        
        logger.info(f"Report saved: {filepath}")
        return filepath
    
    def print_analysis(self, result: AnalysisResult):
        """Pretty print analysis results"""
        print("\n" + "="*80)
        print("🔬 AI-POWERED DEEPFAKE DETECTION REPORT")
        print("="*80)
        
        print(f"\n📁 File: {os.path.basename(result.file_path)}")
        print(f"🔑 Hash: {result.file_hash[:16]}...")
        print(f"⏱️  Analysis Time: {result.analysis_time_ms}ms")
        
        # Verdict with emoji
        verdict_emoji = {
            'ai_generated': '🤖',
            'authentic': '✅',
            'suspicious': '⚠️',
            'inconclusive': '❓'
        }
        emoji = verdict_emoji.get(result.verdict, '❓')
        
        print(f"\n{emoji} VERDICT: {result.verdict.upper()}")
        print(f"📊 Confidence: {result.confidence_score}%")
        print(f"⚠️  Risk Level: {result.risk_level.upper()}")
        
        if result.ai_model_detected:
            print(f"🤖 AI Model: {result.ai_model_detected}")
        
        if result.camera_detected:
            print(f"📷 Camera: {result.camera_detected}")
        
        print(f"\n💡 REASONING:")
        print(f"   {result.reasoning}")
        
        print(f"\n🔍 KEY FINDINGS:")
        for finding in result.key_findings:
            print(f"   • {finding}")
        
        print(f"\n📋 METADATA SUMMARY:")
        for key, value in result.metadata_summary.items():
            if value:
                print(f"   {key}: {value}")
        
        print(f"\n⏰ Analyzed: {result.timestamp}")
        print("="*80 + "\n")


def main():
    """Example usage"""
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found in environment variables!")
        print("\nSetup instructions:")
        print("1. Create a .env file in the same directory")
        print("2. Add: OPENAI_API_KEY=your_api_key_here")
        print("3. Or set environment variable: export OPENAI_API_KEY=your_key")
        print("\nGet your API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize system
    detector = DeepfakeDetectionSystem()
    
    # Analyze image
    image_path = "openai.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Update the image_path variable with your image file")
        return
    
    try:
        print(f"🔬 Analyzing: {image_path}")
        print("⏳ Please wait...\n")
        
        # Run analysis
        result = detector.analyze_image(image_path)
        
        # Print results
        detector.print_analysis(result)
        
        # Save report
        report_path = detector.save_report(result)
        print(f"📄 Detailed report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

