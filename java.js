const track = document.getElementById("image-track");

window.onmousedown = e => {
	track.dataset.mouseDownAt = e.clientX;
}



window.onmousemove = e => {

	if (track.dataset.mouseDownAt==="0")return
	const mouseDelta = parseFloat(track.dataset.mouseDownAt)- e.clientX , 
	maxDelta =window.innerWidth;
	const percentage = (mouseDelta/maxDelta)*-100;
	nextPercentage = parseFloat(track.dataset.prevPercentage)+percentage;
	if (nextPercentage>0){
		nextPercentage=0;
	}
	if (nextPercentage<-100){
		nextPercentage=-100;
	}

	
    track.dataset.percentage = nextPercentage;
    track.style.transform = "translate(" + nextPercentage + "%, -50%)";
    

	

	for(const image of track.getElementsByClassName("image")){
		image.animate({objectPosition:nextPercentage +100+ "% 50%" },{duration:1200,fill:"forwards"});
	}

}


window.onmouseup = e => {
	track.dataset.mouseDownAt = "0";
	if(track.dataset.percentage===undefined){
		track.dataset.percentage=0;
	}

	track.dataset.prevPercentage = track.dataset.percentage;
    
}



function FullView(src, projectId) {
    const image = Array.from(document.getElementsByClassName("image")).find(img => img.src.includes(src));
    const modal = document.getElementById("modal");
    const modalImage = document.getElementById("modal-image");
    const modalDescription = document.getElementById("modal-description");
    const projectTitle = document.getElementById("project-title");
    const projectDescription = document.getElementById("project-description");

    if (image) {
        modal.style.display = "block";
        modalImage.src = src;
        if (projectDetails[projectId]) {
            projectTitle.textContent = projectDetails[projectId].title;
            projectDescription.innerHTML = projectDetails[projectId].description;
        }
    }
}

// Close the modal logic
const span = document.getElementsByClassName("close")[0];
span.onclick = function() {
    modal.style.display = "none";
}
window.onclick = function(event) {
    if (event.target === modal) {
        modal.style.display = "none";
    }
}





const projectDetails = {
    fusion: {
        title: "Content Fusion Web App",
        description: `
            <h3>1. Project Aim</h3>
            <p>The primary goal of the Content Fusion Web App is to provide a solution for content creators, particularly social media influencers, who often struggle with idea generation after producing a significant number of posts or videos. The app aims to stimulate creativity by fusing two distinct ideas submitted by the user to generate new, unique content suggestions. This supports creators in maintaining engagement with their audience by continually offering fresh and innovative content.</p>
            
            <h3>2. Concept and Idea</h3>
            <p>The core concept of the Content Fusion Web App revolves around leveraging artificial intelligence to assist in content ideation. Users input two separate ideas, and through the integration of AI, specifically a language model, the app generates a synthesis of these ideas into several new content proposals. Users can then select their preferred idea and further use the app to generate visual and video content, enhancing the conceptualization and planning process of content creation.</p>
            
            <h3>3. Workflow and Architecture</h3>
            <p>The workflow of the app involves several stages:</p>
            <ul>
                <li><strong>Idea Input:</strong> Users enter two distinct ideas.</li>
                <li><strong>Idea Fusion:</strong> Using OpenAI's ChatGPT API, the app fuses the ideas to generate new content suggestions.</li>
                <li><strong>Idea Selection:</strong> Users select their favorite idea from the generated options.</li>
                <li><strong>Image Prompt Generation:</strong> For the selected idea, the app, again using ChatGPT, generates image prompts that capture the essence of the idea.</li>
                <li><strong>Image Generation:</strong> These prompts are fed into DALL-E to create relevant images.</li>
                <li><strong>Video Compilation:</strong> Using the images, a slideshow video is created with captions derived from the selected idea, providing a visual explanation of the concept.</li>
            </ul>

			<p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/AI-Content-Fusion" target="_blank">GitHub Repository</a>
            
            <p><strong>Watch the demo video on YouTube:</strong></p>
            <a href="https://www.youtube.com/watch?v=hxaa5btqmfE" target="_blank">YouTube Demo Video</a>
			
        `
    },
    
    rhythmpass : {
        title: "Rhythm-Password",
        description: `
            <h3>Overview</h3>
            <p>This project introduces a novel security system that integrates gesture and rhythm recognition for enhanced access control. Utilizing the Arduino Nano 33 BLE Sense, it features a dual-layered security approach combining a gesture-based lock with a rhythm-based lock, leveraging machine learning for improved accuracy and user experience.</p>
            <h3>Features</h3>
            <ul>
                <li><strong>Gesture Lock:</strong> Uses the Arduino's APDS-9960 sensor for gesture detection, allowing hand gestures as a unique access code.</li>
                <li><strong>Rhythm Lock:</strong> Incorporates keystroke dynamics analyzed through machine learning, adding a layer of security by recognizing the unique rhythm of key presses.</li>
                <li><strong>Machine Learning:</strong> Employs TensorFlow/Keras Sequential models optimized for embedded systems, providing real-time processing capabilities.</li>
            </ul>   
            <h3>Future Work</h3>
            <ul>
                <li>Simplify the rhythm lock setup process based on user feedback.</li>
                <li>Explore the integration of AI-driven rhythm assignment for ease of use.</li>
                <li>Extend the application scope to include residential security, personal device access, and corporate security solutions.</li>
            </ul>
            <h3>Links</h3>
            <a href="https://drive.google.com/file/d/1NdiChjM--mscFrfm1GLwdCWAmSlvJz8P/view?usp=sharing" target="_blank">Watch the Demo Video</a>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Rhythm-Password-KeyStroke-Dynamics/tree/main" target="_blank">GitHub Repository</a>
        `
    },

    voice_recognition : {
        title: "Voice Recognition for Mental Health",
        description: `
            <h3>Overview</h3>
            <p>This project utilizes the Arduino Nano 33 BLE Sense to develop a voice recognition system that identifies keywords associated with absolutist language, commonly found in mental health diagnostics. The aim is to monitor potential language markers of mental health states.</p>   
            <h3>Data Collection & Dataset</h3>
            <ul>
                <li>Sources: Personal recordings and student contributions.</li>
                <li>Keywords: 6 (including silence), with a total of 2531 samples split 80% for training and 20% for testing.</li>
                <li>Stored in '.wav' format, utilizing the Harvard speech recording tool and the Simple Speech Command dataset for training.</li>
            </ul>
    
            <h3>System Design and Training</h3>
            <ul>
                <li>Tool: Edge Impulse with 1D CNNs architecture focusing on audio processing.</li>
                <li>Training involved 100 cycles with an accuracy of 96.8% and a quick inference time of 3ms.</li>
            </ul>
    
            <h3>Demo & Results</h3>
            <p>Highly accurate system capable of real-time keyword detection. See it in action:</p>
            <a href="https://drive.google.com/file/d/1bSiBapBsx0R0te8azuRlGFuSWncwjrBX/view?usp=sharing" target="_blank">Demo Video</a>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Voice-Recognition-for-mental-health-using-Deep-Learning" target="_blank">GitHub Repository</a>

        `
    },

    android_malware : {
        title: "Analyzing Malicious Android Apps",
        description: `
            <h3>Overview</h3>
            <p>This project explores using machine learning techniques to automatically detect malicious Android applications, aiming to enhance security measures against evolving Android malware threats.</p>
            <h3>Features & Data Preparation</h3>
            <ul>
                <li><strong>Malware Detection:</strong> Uses machine learning to distinguish between benign and malicious apps.</li>
                <li><strong>Feature Extraction:</strong> Extracts features from APK files such as permissions and activities using Python scripts and APKtool.</li>
                <li><strong>Data Collection:</strong> Compiled 1000 APK samples, with a mix of benign and malware instances for analysis.</li>
            </ul>
    
            <h3>Machine Learning Process</h3>
            <p>Evaluated 16 machine learning models, such as Random Forest and SGD Classifier, based on their F1 scores to determine the most effective models for identifying malware.</p>
    
            <h3>Results & Achievements</h3>
            <ul>
                <li><strong>Best Performance:</strong> Permissions used by apps were the most significant feature for malware detection, with the SGD Classifier achieving the best F1 score of 0.86.</li>
            </ul>
            <h3>Future Directions</h3>
            <p>Plans to expand feature extraction techniques, increase the dataset size, and explore deep learning approaches for improved malware detection accuracy.</p>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Android-Malware-App-Detection-using-Machine-Learning" target="_blank">GitHub Repository</a>
        `
    },

    ensemble_method : {
        title: "Sentiment Analysis using Machine Learning and Deep Learning",
        description: `
            <h3>Overview</h3>
            <p>This project dives into sentiment analysis to determine the polarity of text data from social media, focusing on Twitter due to its vast, real-time user-generated content. We utilize both machine learning and deep learning techniques to enhance automated sentiment analysis.</p>
            <h3>Problem Statement & Datasets</h3>
            <p>Our goal is to analyze Twitter data to categorize sentiments as positive, neutral, or negative. We used the SemEval-2014 Task 9 dataset for binary and fine-grained sentiment classification of tweets.</p>
    
            <h3>Machine Learning & Deep Learning Algorithms</h3>
            <ul>
                <li><strong>ML Algorithms:</strong> Logistic Regression, SVM, Naive Bayes, Decision Trees, Random Forest.</li>
                <li><strong>DL Algorithms:</strong> CNN, RNN (including LSTM, GRU), and RoBERTa.</li>
            </ul>
    
            <h3>System Architecture & User Interface</h3>
            <p>Integrates a Flask backend with a GUI that allows users to input text and receive sentiment analyses. This setup provides a dynamic interface for real-time sentiment evaluation.</p>
    
            <h3>Evaluation Metrics</h3>
            <ul>
                <li>Metrics include Accuracy, Precision, Recall, and F1-Score, highlighting the superior performance of deep learning models, particularly RoBERTa.</li>
            </ul>
    
            <h3>Conclusion & Future Directions</h3>
            <p>The project underscores the effectiveness of deep learning in sentiment analysis on Twitter data, suggesting potential expansions to more datasets and sophisticated models.</p>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Sentiment-Analysis-using-Ensemble-Machine-Learning-Methods" target="_blank">GitHub Repository</a>
        `
    },

    reinforcement : {
        title: "Beyond-Demonstration: Enhancing Human-Robot Collaboration",
        description: `
            <h3>Introduction</h3>
            <p>This project advances Human-Robot Collaboration by refining the Learning from Demonstration (LfD) approach. Focusing on the D-REX algorithm, which utilizes ranked demonstrations for Inverse Reinforcement Learning (IRL), we propose enhancements to minimize performance variability and optimize rewards.</p>
    
            <h3>Background & Approach</h3>
            <p>We build on Imitation Learning and IRL, enhancing the D-REX algorithm that processes expert trajectories with varying noise levels. Our improvements include a more stable preference model, fixed horizon rollouts, and a streamlined single reward network.</p>
    
            <h3>Implementation & Technologies</h3>
            <p>Utilizing Python, TensorFlow, and stable_baselines3, we implemented the IRL algorithm to train policies that surpass the demonstrators' performances. Our system uses techniques like noise injection and preference-based training to optimize the learning process.</p>
    
            <h3>Results & Changes to Baseline</h3>
            <p>Significant advancements include implementing a Luce preference model, mixed sampling, and fixed horizon rollouts. These adaptations have led to demonstrably superior performance in environments like HalfCheetah-v3 and comparable outcomes in Hopper-v3.</p>
    
            <h3>Conclusion & Future Work</h3>
            <p>Our project marks progress in IRL by enhancing the D-REX algorithm, showing promising results in complex tasks. Future directions involve expanding to more complex environments and integrating advanced IRL techniques.</p>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Reinforcement-Learning-DREX-" target="_blank">GitHub Repository</a>
        `
    },

    posture_detection : {
        title: "Embedded Machine Learning for Posture Detection",
        description: `
            <h3>Overview</h3>
            <p>This project focuses on designing a posture detection system that utilizes sensor data from Arduino to distinguish between various postures. Leveraging TensorFlow and the Edge Impulse platform, this system demonstrates the integration of machine learning into a microcontroller environment, achieving an impressive accuracy of approximately 88.65% on test data.</p>
    
            <h3>Features & Data Handling</h3>
            <ul>
                <li><strong>Data Collection:</strong> Captured using accelerometer, gyroscope, and magnetometer data from Arduino sensors, stored in CSV format.</li>
                <li><strong>Machine Learning Model:</strong> Uses a CNN architecture tailored for time-series sensor data, demonstrating the capability of real-time posture predictions.</li>
            </ul>
    
            <h3>Machine Learning Process</h3>
            <p>Implemented a CNN with layers suited for sensor data analysis, trained using the Adam optimizer for 50 epochs, achieving a test accuracy of approximately 88.65%.</p>
    
            <h3>Results & Application</h3>
            <p>The system is integrated into Arduino for real-time posture detection, with practical applications in health monitoring and ergonomic equipment. Achieved around 70% accuracy in real-world conditions.</p>
    
            <h3>DEMO & Setup</h3>
            <p>Explore the model's capabilities through a live demo. Setup involves TensorFlow installation and model training via Google Colab. Detailed instructions are provided in the notebooks.</p>
            <a href="https://drive.google.com/file/d/1VVS8PMOueLMXs5onEnYGI6CSPbOLqEiP/view?usp=sharing" target="_blank">Demo Video</a>
            <p><strong>Explore the code on GitHub:</strong></p>
            <a href="https://github.com/pranit1812/Posture-Detection-using-Machine-Learning-" target="_blank">GitHub Repository</a>
        `
    },

    cnn : {
        title: "Comparative Analysis of CNN Designs for CIFAR-10 Classification",
        description: `
            <h3>Overview</h3>
            <p>This project performs a comparative analysis of various Convolutional Neural Network (CNN) architectures for object classification on the CIFAR-10 dataset, exploring different model designs from basic CNNs to advanced architectures like MobileNetV2.</p>
            <h3>Experimentation & Motivation</h3>
            <p>The motivation for this project is to understand how architectural decisions affect the performance of image classification tasks in deep learning. The study progresses through several design phases:</p>
            <ul>
                <li><strong>Phase 1:</strong> Basic CNN Model without pooling layers.</li>
                <li><strong>Phase 2:</strong> Enhanced model with pooling layers for improved efficiency.</li>
                <li><strong>Phase 3:</strong> Augmented model with dropout layers to combat overfitting.</li>
                <li><strong>Phase 4:</strong> Implementation of MobileNetV2, a lightweight and efficient architecture.</li>
            </ul>
    
            <h3>Framework & Tools</h3>
            <p>The experiments are conducted using TensorFlow on Google Colab with the CIFAR-10 dataset, split into training, validation, and testing sets.</p>
    
            <h3>Results & Analysis</h3>
            <p>The study reveals varying performance across different phases:</p>
            <ul>
                <li><strong>Phase 1 Accuracy:</strong> 50.55% - Basic feature extraction capabilities without pooling layers.</li>
                <li><strong>Phase 2 Accuracy:</strong> 71.61% - Significant improvement with the introduction of pooling layers.</li>
                <li><strong>Phase 3 Accuracy:</strong> 70.40% - Slightly reduced accuracy, potentially due to over-tuning of dropout layers.</li>
                <li><strong>Phase 4 Accuracy:</strong> 60.68% - Performance of MobileNetV2 indicates the need for fine-tuning to optimize results on CIFAR-10.</li>
            </ul>
    
            <h3>Conclusion</h3>
            <p>The addition of pooling layers in Phase 2 yielded the best performance, emphasizing the balance between model complexity and efficiency. Future work will explore deeper architectures, optimize dropout rates, and focus on hyperparameter tuning, particularly considering the challenges of overfitting.</p>
            <p><strong>Explore the code on Github</strong></p>
            <a href="https://github.com/pranit1812/Comparative-Analysis-of-CNN-Designs-for-CIFAR-10-Object-Classification" target="_blank">GitHub Repository</a>
        `
    }
    
    
    
    
    
    

    



	

}
