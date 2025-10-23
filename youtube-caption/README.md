# Azure Speech and Translator Setup Guide

## Step 1: Create an Azure Account
Go to the [Azure Portal](https://portal.azure.com/) and create a new account if you don’t already have one.

---

## Step 2: Create a Resource Group
Create a new **Resource Group** to organize your Azure services.  
![Resource Group Screenshot](<Screenshot from 2025-10-23 06-38-25.png>)

---

## Step 3: Create a Speech Service
Create a **Speech Service** and select the **Free (F0)** subscription tier.  
![Speech Service Screenshot](<Screenshot from 2025-10-23 00-18-15.png>)

---

## Step 4: Create a Translator Service
Create a **Translator** service and select the **Free (F0)** subscription tier.  
![Translator Service Screenshot](<Screenshot from 2025-10-23 00-17-42.png>)

---

## Step 5: Get the Keys and Endpoints
Open both the **Speech Service** and **Translator** resources.  
Navigate to **Keys and Endpoints**, then copy the **Key** and **Endpoint** values.  
![Speech Keys Screenshot](<Screenshot from 2025-10-23 00-18-32.png>)  
![Translator Keys Screenshot](<Screenshot from 2025-10-23 00-18-56.png>)

---

## Step 6: Configure Environment Variables
In the **backend** folder, create a `.env` file and add the copied keys and endpoints.  
Example:
```env
SPEECH_KEY=your_speech_service_key
SPEECH_ENDPOINT=your_speech_service_endpoint
TRANSLATOR_KEY=your_translator_key
TRANSLATOR_ENDPOINT=your_translator_endpoint
```

---

## Step 7: Run the Backend
Navigate to the **backend** folder and run:
```bash
npm install
npm start
```

---

## Step 8: Run the Frontend
Navigate to the **frontend** folder and run:
```bash
npm install
npm run dev
```

---

✅ You’ve successfully set up your Azure Speech and Translator services and configured the project!

