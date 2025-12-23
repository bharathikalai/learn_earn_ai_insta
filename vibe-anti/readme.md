# Building Web Apps with Antigravity

> A Complete Guide to AI-Powered Development

Learn how to leverage Antigravity's vibe coding tool with Claude Opus to build full-stack applications efficiently.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Phase 1: Setting Up Antigravity](#phase-1-setting-up-antigravity)
- [Phase 2: MongoDB Database Setup](#phase-2-mongodb-database-setup)
- [Phase 3: Preparing Your Application Prompt](#phase-3-preparing-your-application-prompt)
- [Phase 4: Building with Antigravity](#phase-4-building-with-antigravity)
- [Post-Generation Steps](#post-generation-steps)
- [Tips for Success](#tips-for-success)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Introduction

Antigravity is an AI-powered coding tool that revolutionizes web application development by allowing you to describe your app in natural language. Combined with MongoDB for database management and Claude Opus for intelligent code generation, you can build production-ready applications in a fraction of the time.

This guide will walk you through the complete setup process and best practices for building web applications using the Antigravity platform.

---

## ✅ Prerequisites

Before you begin, ensure you have:

- A computer running Windows, macOS, or Linux
- Stable internet connection
- Basic understanding of web development concepts (helpful but not required)
- Email account for registration

---

## Phase 1: Setting Up Antigravity

### Step 1: Create Your Antigravity Account

1. Visit the official Antigravity website
2. Click on "Sign Up" or "Get Started"
3. Enter your email address and create a strong password
4. Verify your email address through the confirmation link
5. **Activate your 1-month free trial** - This gives you full access to all features including the Opus model

> 💡 **Tip:** Make sure to use a valid email address as you'll receive important notifications and updates about your projects.

### Step 2: Download and Install Antigravity

1. Navigate to the Downloads section in your Antigravity dashboard
2. Select your operating system (Windows, macOS, or Linux)
3. Download the installer package
4. Run the installer and follow the on-screen instructions
5. Launch Antigravity and sign in with your credentials

> ⚠️ **Important:** Ensure you have administrator privileges on your computer to complete the installation successfully.

---

## Phase 2: MongoDB Database Setup

### Step 3: Create MongoDB Atlas Account

MongoDB Atlas is a cloud-based database service that provides a free tier perfect for development and small applications.

1. Go to [mongodb.com/cloud/atlas](https://mongodb.com/cloud/atlas)
2. Click "Try Free" or "Sign Up"
3. Register using your email, Google, or GitHub account
4. Complete the registration survey (optional but recommended)

### Step 4: Create Your Database Cluster

1. After logging in, click "Build a Database"
2. Select the **FREE** tier (M0 Sandbox)
3. Choose a cloud provider (AWS, Google Cloud, or Azure)
4. Select a region closest to your location for better performance
5. Give your cluster a meaningful name (e.g., "MyAppCluster")
6. Click "Create Cluster" and wait for deployment (2-5 minutes)

> 💡 **Tip:** The free tier includes 512MB of storage, which is sufficient for most development projects and small applications.

### Step 5: Get Database Credentials

1. Once your cluster is ready, click "Connect"
2. Add your current IP address to the whitelist (or use `0.0.0.0/0` for development)
3. Create a database user with username and password
4. Choose "Connect your application"
5. Select "Node.js" as your driver and the latest version
6. **Copy the connection string** - it will look like:
   ```
   mongodb+srv://username:password@cluster.mongodb.net/
   ```
7. Save this connection string securely - you'll need it for Antigravity

> ⚠️ **Security Note:** Never share your connection string publicly. Replace the placeholder password with your actual database password. Store credentials in environment variables in production.

---

## Phase 3: Preparing Your Application Prompt

### Step 6: Use ChatGPT to Polish Your Prompt

A well-structured prompt is crucial for Antigravity to understand your requirements accurately.

1. Open ChatGPT or your preferred AI assistant
2. Describe your application idea in detail
3. Ask the AI to help you create a structured development prompt
4. **Key requirement:** Specify that frontend and backend should be in **separate folders**

#### Example Prompt Structure:

```
Create a task management web application with the following structure:

PROJECT STRUCTURE:
- /frontend - React-based user interface
- /backend - Node.js/Express API server

FEATURES:
- User authentication (register, login, logout)
- Create, read, update, delete tasks
- Task categories and priorities
- Due date tracking
- Responsive design for mobile and desktop

DATABASE SCHEMA:
- Users collection (username, email, password, createdAt)
- Tasks collection (title, description, category, priority, dueDate, userId, completed)

TECHNOLOGY STACK:
- Frontend: React, Tailwind CSS
- Backend: Node.js, Express
- Database: MongoDB
- Authentication: JWT tokens
```

> 💡 **Best Practice:** Separating frontend and backend into different folders helps the AI understand your architecture better and generates cleaner, more maintainable code. This also makes deployment easier.

---

## Phase 4: Building with Antigravity

### Step 7: Configure Antigravity Project

1. Open Antigravity on your computer
2. Create a new project and give it a descriptive name
3. Paste your polished application prompt into Antigravity
4. **Provide your MongoDB connection string** in the database configuration section
5. This allows the AI to understand your database structure and generate appropriate code

> ⚠️ **Important:** Sharing your database connection with Antigravity allows it to generate schema-aware code. Ensure you're using a development database, not a production one.

### Step 8: Select AI Model and Mode

#### Choose Opus Mode

In the model selection dropdown, choose **Opus Mode**. Opus is Claude's most powerful model, offering:

- Superior code quality and architecture
- Better understanding of complex requirements
- More accurate database schema generation
- Enhanced error handling and edge case consideration

#### Select Planning Mode

Choose **Planning Mode** for your generation approach. Planning Mode ensures the AI:

- Creates a detailed project blueprint first
- Organizes code into logical components
- Follows best practices and design patterns
- Generates scalable and maintainable code

### Step 9: Generate Your Application

1. Review your configuration one final time
2. Click **"Start Create"** or "Generate"
3. Antigravity will begin building your application:
   - Creating project structure
   - Generating frontend components
   - Building backend API endpoints
   - Setting up database models and connections
   - Adding authentication and security features
4. Monitor the progress in the Antigravity interface
5. The generation process typically takes 5-15 minutes depending on complexity

---

## Post-Generation Steps

### Step 10: Review and Test

1. Once generation is complete, review the generated code structure
2. Install dependencies for both frontend and backend:
   ```bash
   cd frontend && npm install
   cd ../backend && npm install
   ```
3. Configure environment variables with your MongoDB connection string:
   ```bash
   # backend/.env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/myapp
   PORT=5000
   JWT_SECRET=your_secret_key_here
   ```
4. Start the development servers:
   ```bash
   # Terminal 1 - Backend
   cd backend
   npm run dev
   
   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```
5. Test all features and functionality
6. Make any necessary adjustments or refinements

---

## 🎯 Tips for Success

- **Be specific** in your prompts - the more detail you provide, the better the output
- **Start simple** - build core features first, then iterate and add complexity
- **Test incrementally** - verify each feature works before moving to the next
- **Keep credentials secure** - use environment variables and never commit secrets to version control
- **Leverage the AI** - if something doesn't work, describe the issue and ask Antigravity to fix it
- **Document your changes** - maintain notes on customizations you make to the generated code

---

## 🔧 Troubleshooting

### Common Issues

**Issue: MongoDB connection fails**
- Solution: Verify your IP address is whitelisted in MongoDB Atlas
- Check that your connection string password doesn't contain special characters that need URL encoding

**Issue: Frontend can't connect to backend**
- Solution: Ensure CORS is properly configured in your backend
- Verify both servers are running on the correct ports

**Issue: Generated code has errors**
- Solution: Check that all dependencies are installed
- Review the error messages and ask Antigravity to fix specific issues

**Issue: Database queries not working**
- Solution: Verify your MongoDB connection string is correct
- Check that database models match your schema requirements

---

## 📚 Additional Resources

- [MongoDB Documentation](https://docs.mongodb.com/)
- [Antigravity Documentation](https://antigravity.com/docs)
- [Claude AI Best Practices](https://docs.anthropic.com/)
- [React Documentation](https://react.dev/)
- [Node.js Documentation](https://nodejs.org/docs/)

---

## 📝 License

This guide is provided as-is for educational purposes.

---

## 🤝 Contributing

If you have suggestions for improving this guide, please feel free to contribute or reach out.

---

## ⭐ Conclusion

Antigravity with Claude Opus revolutionizes web development by transforming your ideas into functional applications rapidly. By following this guide and leveraging the planning mode with separated frontend and backend architecture, you're set up for success in building scalable, maintainable web applications.

**Happy Coding! 🚀**

---

*Last Updated: December 2024*