import express from "express";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import fs from "fs";
import axios from "axios";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

ffmpeg.setFfmpegPath(ffmpegPath);

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const app = express();

app.use(cors({ 
  origin: true,
  credentials: true
}));

app.use(express.json());

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/')) cb(null, true);
    else cb(new Error('Not a video file!'), false);
  },
  limits: { fileSize: 100 * 1024 * 1024 }
});

const speechKey = process.env.AZURE_SPEECH_KEY;
const speechRegion = process.env.AZURE_SPEECH_REGION;
const translatorKey = process.env.AZURE_TRANSLATOR_KEY;

if (!speechKey || !speechRegion) {
  console.error("❌ ERROR: Azure credentials not configured!");
  process.exit(1);
}

console.log('✓ Azure Speech Region:', speechRegion);
console.log('✓ Speech Key configured');

// Extract text from video
async function extractTextFromVideo(videoPath) {
  const audioPath = `${videoPath}.wav`;

  try {
    // Step 1: Extract audio
    console.log('📹 Extracting audio from video...');
    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .output(audioPath)
        .audioCodec('pcm_s16le')
        .audioFrequency(16000)
        .audioChannels(1)
        .toFormat('wav')
        .on("end", () => {
          console.log('✓ Audio extracted successfully');
          resolve();
        })
        .on("error", (err) => {
          console.error('❌ FFmpeg error:', err);
          reject(err);
        })
        .run();
    });

    // Step 2: Read audio file
    console.log('📖 Reading audio file...');
    const audioBuffer = fs.readFileSync(audioPath);
    const pcmData = audioBuffer.slice(44); // Skip WAV header
    
    console.log(`✓ Audio size: ${audioBuffer.length} bytes`);
    console.log(`✓ PCM data: ${pcmData.length} bytes`);

    // Step 3: Setup Azure Speech Recognition
    console.log('🎤 Setting up Azure Speech Recognition...');
    
    const pushStream = sdk.AudioInputStream.createPushStream();
    pushStream.write(pcmData);
    pushStream.close();

    const speechConfig = sdk.SpeechConfig.fromSubscription(speechKey, speechRegion);
    speechConfig.speechRecognitionLanguage = "ta-IN";

    const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
    const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

    // Step 4: Recognize speech
    console.log('🔊 Starting speech recognition...');
    
    const transcription = await new Promise((resolve, reject) => {
      let fullText = "";
      let segmentCount = 0;
      let isResolved = false;

      recognizer.recognizing = (s, e) => {
        if (e.result.text) {
          console.log(`⏳ Recognizing: ${e.result.text}`);
        }
      };

      recognizer.recognized = (s, e) => {
        if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
          segmentCount++;
          console.log(`✓ Recognized #${segmentCount}: ${e.result.text}`);
          fullText += e.result.text + " ";
        } else if (e.result.reason === sdk.ResultReason.NoMatch) {
          console.log('⚠️  No speech in this segment');
        }
      };

      recognizer.canceled = (s, e) => {
        console.log(`📢 Cancellation reason: ${sdk.CancellationReason[e.reason]}`);
        
        if (!isResolved) {
          isResolved = true;
          
          // EndOfStream means audio finished - this is SUCCESS
          if (e.reason === sdk.CancellationReason.EndOfStream || 
              e.reason === sdk.CancellationReason.CancelledByUser) {
            
            recognizer.close();
            
            if (segmentCount > 0 && fullText.trim()) {
              console.log(`✓ Recognition completed with ${segmentCount} segments`);
              resolve(fullText.trim());
            } else {
              reject(new Error('No speech detected in video'));
            }
          } else {
            // Real error
            console.error(`❌ Recognition error: ${e.errorDetails}`);
            recognizer.close();
            reject(new Error(e.errorDetails || 'Recognition failed'));
          }
        }
      };

      recognizer.sessionStopped = (s, e) => {
        console.log('🛑 Session stopped');
        
        if (!isResolved) {
          isResolved = true;
          recognizer.close();
          
          if (segmentCount > 0 && fullText.trim()) {
            console.log(`✓ Final text with ${segmentCount} segments`);
            resolve(fullText.trim());
          } else {
            reject(new Error('No speech detected in video'));
          }
        }
      };

      recognizer.startContinuousRecognitionAsync(
        () => {
          console.log('✓ Recognition started');
        },
        (err) => {
          console.error(`❌ Failed to start: ${err}`);
          if (!isResolved) {
            isResolved = true;
            recognizer.close();
            reject(new Error(`Failed to start recognition: ${err}`));
          }
        }
      );
    });

    // Cleanup
    if (fs.existsSync(audioPath)) {
      fs.unlinkSync(audioPath);
      console.log('🗑️  Cleaned up audio file');
    }

    console.log('✅ Final transcription:', transcription);
    return transcription;

  } catch (error) {
    // Cleanup on error
    if (fs.existsSync(audioPath)) {
      fs.unlinkSync(audioPath);
    }
    throw error;
  }
}

// Upload endpoint
app.post("/api/upload", upload.single("video"), async (req, res) => {
  let inputPath;

  try {
    if (!req.file) {
      return res.status(400).json({ error: "No video uploaded" });
    }

    inputPath = req.file.path;
    console.log('\n' + '='.repeat(60));
    console.log(`📦 Processing: ${req.file.originalname}`);
    console.log('='.repeat(60));

    const tamilText = await extractTextFromVideo(inputPath);

    console.log('='.repeat(60));
    console.log('✅ SUCCESS!');
    console.log('Tamil text:', tamilText);
    console.log('='.repeat(60) + '\n');

    // Cleanup video
    if (fs.existsSync(inputPath)) {
      fs.unlinkSync(inputPath);
    }

    res.json({
      success: true,
      text: tamilText,
      language: "ta"
    });

  } catch (error) {
    console.log('='.repeat(60));
    console.error('❌ ERROR:', error.message);
    console.log('='.repeat(60) + '\n');

    if (inputPath && fs.existsSync(inputPath)) {
      fs.unlinkSync(inputPath);
    }

    res.status(500).json({
      error: "Error processing video",
      details: error.message
    });
  }
});

// Translate endpoint
app.post("/api/translate", async (req, res) => {
  try {
    const { text, fromLang, toLang } = req.body;

    if (!text) {
      return res.status(400).json({ error: "No text provided" });
    }

    if (!translatorKey) {
      return res.status(400).json({ error: "Translator not configured" });
    }

    console.log(`\n🌍 Translating: ${fromLang} → ${toLang}`);

    const endpoint = `https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=${fromLang}&to=${toLang}`;

    const response = await axios.post(
      endpoint,
      [{ text }],
      {
        headers: {
          "Ocp-Apim-Subscription-Key": translatorKey,
          "Ocp-Apim-Subscription-Region": speechRegion,
          "Content-Type": "application/json"
        }
      }
    );

    const translatedText = response.data[0]?.translations[0]?.text || text;
    console.log(`✓ Translation: ${translatedText}\n`);

    res.json({
      success: true,
      text: translatedText,
      language: toLang
    });

  } catch (error) {
    console.error('❌ Translation error:', error.response?.data || error.message);
    res.status(500).json({
      error: "Translation failed",
      details: error.response?.data?.error?.message || error.message
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log('\n' + '='.repeat(60));
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log('='.repeat(60) + '\n');
});