from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global ML Model Configuration
model = None
feature_names = ['Math', 'Science', 'Art', 'Communication', 'Logic', 'Coding', 'Speaking', 'Management']

# Career Knowledge Database for Roadmap and Skills
# This registry maps careers to specific learning paths
CAREER_KNOWLEDGE = {
    "Software Developer": {
        "degree": "B.Tech/BCA/MCA in Computer Science or related field.",
        "salary": {
            "fresher": "4 – 12 LPA",
            "mid": "12 – 25 LPA",
            "senior": "25+ LPA"
        },
        "roadmap": [
            "Learn Programming Basics",
            "Master DSA",
            "Learn OOP Concepts",
            "Build Real Projects",
            "System Design Basics"
        ],
        "skills": [
            "Problem Solving",
            "Data Structures",
            "OOP",
            "Debugging",
            "System Design"
        ],
        "languages": {
            "know": ["C", "C++"],
            "master": ["Java", "Python", "JavaScript"]
        },
        "certifications": [
            "Oracle Java Certification",
            "AWS Developer Associate",
            "Microsoft Azure Developer"
        ],
        "future_scope": "Very High – Every company needs software engineers.",
        "job_roles": [
            "Backend Developer",
            "Frontend Developer",
            "Mobile Developer",
            "System Engineer"
        ]
    },
    "Data Scientist": {
    "degree": "B.Tech/BSc in Computer Science, Data Science, Statistics, Mathematics or related field. MCA/MSc is an added advantage.",
    "salary": {
        "fresher": "6 – 15 LPA",
        "mid": "15 – 30 LPA",
        "senior": "30+ LPA"
    },
    "roadmap": [
        "Learn Statistics & Probability",
        "Master Python Programming",
        "Data Manipulation (Pandas, NumPy)",
        "Data Visualization (Matplotlib, Seaborn)",
        "Machine Learning Algorithms",
        "Deep Learning (Optional Advanced)",
        "Model Deployment (Flask/FastAPI)"
    ],
    "skills": [
        "Statistical Analysis",
        "Machine Learning",
        "Data Cleaning",
        "Feature Engineering",
        "Model Evaluation",
        "Problem Solving"
    ],
    "languages": {
        "know": ["SQL", "R"],
        "master": ["Python"]
    },
    "certifications": [
        "Google Data Analytics Certificate",
        "IBM Data Science Professional Certificate",
        "AWS Machine Learning Specialty"
    ],
    "future_scope": "Extremely High – AI & ML are transforming every industry.",
    "job_roles": [
        "Data Analyst",
        "Machine Learning Engineer",
        "AI Engineer",
        "Business Intelligence Analyst"
    ]
    },
    "Full Stack Developer": {
    "degree": "B.Tech/BCA/MCA in Computer Science, IT, or related field. Strong project portfolio is highly important.",
    
    "salary": {
        "fresher": "5 – 12 LPA",
        "mid": "12 – 22 LPA",
        "senior": "22 – 40+ LPA"
    },
    
    "roadmap": [
        "Learn HTML, CSS, JavaScript Fundamentals",
        "Master Responsive Design & CSS Frameworks (Bootstrap/Tailwind)",
        "Learn Frontend Framework (React / Angular / Vue)",
        "Backend Development (Node.js / Django / Flask)",
        "Understand REST APIs & Authentication (JWT)",
        "Database Management (MySQL / MongoDB)",
        "Version Control (Git & GitHub)",
        "Deployment (AWS / Vercel / Render / Docker)"
    ],
    
    "skills": [
        "Frontend Development",
        "Backend Development",
        "API Integration",
        "Database Design",
        "Authentication & Security",
        "Debugging",
        "Problem Solving",
        "Version Control"
    ],
    
    "languages": {
        "know": ["HTML", "CSS"],
        "master": ["JavaScript", "Node.js", "Python"]
    },
    
    "certifications": [
        "Meta Frontend Developer Professional Certificate",
        "AWS Cloud Practitioner",
        "Microsoft Full Stack Certification",
        "Google Web Developer Certification"
    ],
    
    "future_scope": "Very High – Startups and product companies highly demand full stack developers.",
    
    "job_roles": [
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Engineer",
        "Web Application Developer",
        "Software Engineer"
    ]
    },
    "Cyber Security Analyst": {
    "degree": "B.Tech in Cyber Security, Computer Science, IT or related field.",
    "salary": {
        "fresher": "5 – 10 LPA",
        "mid": "10 – 20 LPA",
        "senior": "20+ LPA"
    },
    "roadmap": [
        "Learn Networking Fundamentals (TCP/IP)",
        "Understand Linux Systems",
        "Study Cyber Security Basics",
        "Ethical Hacking & Penetration Testing",
        "Security Tools (Wireshark, Metasploit)",
        "Cloud Security Concepts"
    ],
    "skills": [
        "Network Security",
        "Threat Detection",
        "Risk Analysis",
        "Penetration Testing",
        "Encryption",
        "Incident Response"
    ],
    "languages": {
        "know": ["C", "Bash"],
        "master": ["Python", "PowerShell"]
    },
    "certifications": [
        "CEH (Certified Ethical Hacker)",
        "CompTIA Security+",
        "CISSP"
    ],
    "future_scope": "Extremely High – Cyber threats are increasing globally.",
    "job_roles": [
        "Security Analyst",
        "Ethical Hacker",
        "Penetration Tester",
        "Security Consultant"
    ]
    },
    "UI/UX Designer": {
    "degree": "B.Des, BFA, B.Tech in Design/Computer Science, or Professional UI/UX Certification. Strong portfolio matters more than degree.",
    "salary": {
        "fresher": "4 – 8 LPA",
        "mid": "8 – 18 LPA",
        "senior": "18 – 30+ LPA"
    },
    "roadmap": [
        "Understand Design Fundamentals (Color Theory, Typography, Layout)",
        "Learn UX Principles & Human Psychology",
        "Wireframing & Prototyping",
        "Master Tools (Figma, Adobe XD, Sketch)",
        "User Research & Usability Testing",
        "Build Strong Portfolio with Case Studies",
        "Apply for Internships / Freelance Projects"
    ],
    "skills": [
        "Visual Design",
        "User Research",
        "Wireframing",
        "Prototyping",
        "Interaction Design",
        "Usability Testing",
        "Creativity",
        "Communication Skills"
    ],
    "languages": {
        "know": ["HTML", "CSS"],
        "master": ["Figma", "Adobe XD", "Design Systems"]
    },
    "certifications": [
        "Google UX Design Certificate",
        "Interaction Design Foundation Certification",
        "Coursera UI/UX Specialization"
    ],
    "future_scope": "High – Every digital product (Apps, Websites, SaaS) needs great UX.",
    "job_roles": [
        "UI Designer",
        "UX Designer",
        "Product Designer",
        "Interaction Designer",
        "UX Researcher"
    ]
    },
    "DevOps Engineer": {
    "degree": "B.Tech/BCA/MCA in Computer Science or IT.",
    "salary": {
        "fresher": "6 – 12 LPA",
        "mid": "12 – 25 LPA",
        "senior": "25+ LPA"
    },
    "roadmap": [
        "Learn Linux & Bash",
        "Understand Networking",
        "Cloud Platforms (AWS/Azure/GCP)",
        "CI/CD Pipelines",
        "Docker & Kubernetes",
        "Infrastructure as Code (Terraform)"
    ],
    "skills": [
        "Automation",
        "Cloud Computing",
        "Containerization",
        "Monitoring",
        "CI/CD Implementation",
        "Scripting"
    ],
    "languages": {
        "know": ["Python", "Go"],
        "master": ["Bash", "YAML"]
    },
    "certifications": [
        "AWS DevOps Engineer",
        "Docker Certified Associate",
        "Kubernetes Administrator (CKA)"
    ],
    "future_scope": "Very High – Companies are moving toward cloud infrastructure.",
    "job_roles": [
        "Cloud Engineer",
        "Site Reliability Engineer",
        "DevOps Engineer",
        "Automation Engineer"
    ]
}
}

# Default knowledge for careers not explicitly mapped
DEFAULT_KNOWLEDGE = {
    "roadmap": ["Foundational Theory", "Tool Selection", "Practice Projects", "Internships", "Certification"],
    "skills": ["Communication", "Critical Thinking", "Technical Documentation"],
    "languages": {"know": ["Basic Scripting"], "master": ["Domain Specific Tools"]}
}

def train_model():
    """Trains the ML model using the academic career dataset."""
    global model
    try:
        df = pd.read_csv('career_dataset.csv')
        X = df[feature_names]
        y = df['Career']
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model.fit(X, y)
        print("✅ Academic Career ML Model trained successfully!")
    except Exception as e:
        print(f"❌ Error training ML model: {e}")

train_model()

def get_roadmap_response(career):
    """Generates a structured roadmap and skills response."""
    info = CAREER_KNOWLEDGE.get(career, DEFAULT_KNOWLEDGE)
    
    roadmap_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(info['roadmap'])])
    skills_str = ", ".join(info['skills'])
    know_langs = ", ".join(info['languages']['know'])
    master_langs = ", ".join(info['languages']['master'])
    
    return (f"### 🚀 Learning Roadmap for {career}\n"
            f"{roadmap_str}\n\n"
            f"### 🛠️ Key Skills Required\n"
            f"{skills_str}\n\n"
            f"### 💻 Languages\n"
            f"* **Should Know:** {know_langs}\n"
            f"* **Must Master:** {master_langs}")

def generate_offline_chat(user_message, predicted_career):
    msg = user_message.lower()
    info = CAREER_KNOWLEDGE.get(predicted_career, DEFAULT_KNOWLEDGE)

    # DEGREE
    if any(k in msg for k in ["degree", "qualification", "study", "education"]):
        return f"### 🎓 Degree Required for {predicted_career}\n{info.get('degree', 'Relevant degree in the field is recommended.')}"

    # SALARY
    elif any(k in msg for k in ["salary", "pay", "money", "earn", "package"]):
        sal = info.get("salary", {})
        return (f"### 💰 Salary Structure for {predicted_career}\n"
                f"👨‍🎓 Fresher: {sal.get('fresher', '4-8 LPA')}\n"
                f"🧑‍💼 Mid-Level: {sal.get('mid', '8-18 LPA')}\n"
                f"👨‍💻 Senior: {sal.get('senior', '18+ LPA')}")

    # ROADMAP
    elif any(k in msg for k in ["roadmap", "path", "step", "learn"]):
        roadmap = "\n".join([f"{i+1}. {step}" for i, step in enumerate(info['roadmap'])])
        return f"### 🚀 Roadmap for {predicted_career}\n{roadmap}"

    # SKILLS
    elif any(k in msg for k in ["skills", "required", "ability"]):
        skills = ", ".join(info['skills'])
        return f"### 🛠️ Required Skills\n{skills}"

    # LANGUAGES
    elif any(k in msg for k in ["language", "programming"]):
        know = ", ".join(info['languages']['know'])
        master = ", ".join(info['languages']['master'])
        return (f"### 💻 Programming Languages\n"
                f"✔ Should Know: {know}\n"
                f"🔥 Must Master: {master}")

    # CERTIFICATIONS
    elif any(k in msg for k in ["certificate", "certification"]):
        certs = "\n".join([f"- {c}" for c in info.get('certifications', [])])
        return f"### 📜 Recommended Certifications\n{certs}"

    # JOB ROLES
    elif any(k in msg for k in ["role", "job", "position"]):
        roles = "\n".join([f"- {r}" for r in info.get('job_roles', [])])
        return f"### 🧑‍💼 Job Roles in {predicted_career}\n{roles}"

    # FUTURE SCOPE
    elif any(k in msg for k in ["future", "scope", "growth"]):
        return f"### 📈 Future Scope\n{info.get('future_scope', 'High growth potential.')}"

    # DEFAULT
    return (f"I can help you with:\n"
            f"• Degree Required\n"
            f"• Salary Structure\n"
            f"• Roadmap\n"
            f"• Skills Required\n"
            f"• Languages\n"
            f"• Certifications\n"
            f"• Job Roles\n"
            f"• Future Scope\n\n"
            f"Try asking: 'What degree is needed?'")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({'error': 'Model Offline'}), 500
    try:
        data = request.get_json()
        features = [int(data.get(attr.lower(), 5)) for attr in feature_names]
        features_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(features_df)[0]
        
        # Simple Rationale
        explanation = f"Based on your high aptitude in {feature_names[features.index(max(features))]}, you are a perfect fit for {prediction}."
        
        return jsonify({'predicted_career': prediction, 'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        career_context = data.get('career', 'Unknown')
        reply = generate_offline_chat(user_message, career_context)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists('templates'): os.makedirs('templates')
    app.run(debug=True, port=5000)