/**
 * Custom Naive Bayes Classifier for Spam Email Detection
 * Implemented in vanilla JS for client-side execution.
 */

// Basic stop words to optionally filter out (common words that do not help in classification)
const STOP_WORDS = new Set([
    'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]);

/**
 * Clean and tokenize a text string into an array of words
 * @param {string} text - The input email content
 * @param {boolean} removeStopwords - Whether to filter out stop words
 * @returns {string[]} Array of clean word tokens
 */
function tokenize(text, removeStopwords = false) {
    if (!text) return [];
    
    // Normalize string: lowercase, replace newlines/tabs with spaces
    let clean = text.toLowerCase().replace(/[\r\n\t]/g, ' ');
    
    // Keep letters, numbers, and symbols that are highly indicative of spam ($ and %)
    // Replace other punctuation with spaces
    clean = clean.replace(/[^a-z0-9\s$%!]/g, ' ');
    
    // Split on whitespace
    const words = clean.split(/\s+/);
    
    // Filter words: must be length > 1, and optionally not a stop word
    return words.filter(word => {
        const trimmed = word.trim();
        if (trimmed.length <= 1) return false;
        if (removeStopwords && STOP_WORDS.has(trimmed)) return false;
        return true;
    });
}

class NaiveBayesClassifier {
    constructor(removeStopwords = false) {
        this.removeStopwords = removeStopwords;
        this.reset();
    }

    reset() {
        this.spamEmailsCount = 0;
        this.hamEmailsCount = 0;
        this.spamWordCounts = {};
        this.hamWordCounts = {};
        this.totalSpamWords = 0;
        this.totalHamWords = 0;
        this.vocabulary = new Set();
        this.isTrained = false;
    }

    /**
     * Train the classifier with a list of emails
     * @param {Array<{text: string, label: string}>} data - Array of email objects
     */
    train(data) {
        this.reset();
        
        data.forEach(email => {
            const tokens = tokenize(email.text, this.removeStopwords);
            const isSpam = email.label === 'spam';

            if (isSpam) {
                this.spamEmailsCount++;
                tokens.forEach(token => {
                    this.spamWordCounts[token] = (this.spamWordCounts[token] || 0) + 1;
                    this.totalSpamWords++;
                    this.vocabulary.add(token);
                });
            } else {
                this.hamEmailsCount++;
                tokens.forEach(token => {
                    this.hamWordCounts[token] = (this.hamWordCounts[token] || 0) + 1;
                    this.totalHamWords++;
                    this.vocabulary.add(token);
                });
            }
        });

        this.isTrained = true;
    }

    /**
     * Predict whether an email is spam or ham
     * @param {string} text - Email text content
     * @returns {Object} Prediction details
     */
    predict(text) {
        if (!this.isTrained) {
            return { label: 'ham', probability: 0.5, rawSpamLogScore: 0, rawHamLogScore: 0, wordContributions: [] };
        }

        const tokens = tokenize(text, this.removeStopwords);
        if (tokens.length === 0) {
            // Default to equal probability if no words found
            return { label: 'ham', probability: 0.5, rawSpamLogScore: 0, rawHamLogScore: 0, wordContributions: [] };
        }

        const totalEmails = this.spamEmailsCount + this.hamEmailsCount;
        
        // Priors P(Spam) and P(Ham)
        const pSpam = this.spamEmailsCount / totalEmails;
        const pHam = this.hamEmailsCount / totalEmails;
        
        // Start log scores with priors
        let spamLogScore = Math.log(pSpam);
        let hamLogScore = Math.log(pHam);

        const vocabSize = this.vocabulary.size;
        const wordContributions = [];

        tokens.forEach(token => {
            // Count for word in spam and ham (or 0 if never seen)
            const countInSpam = this.spamWordCounts[token] || 0;
            const countInHam = this.hamWordCounts[token] || 0;

            // Laplace smoothing: P(word | Class) = (count + 1) / (total words in class + vocabulary size)
            const pWordGivenSpam = (countInSpam + 1) / (this.totalSpamWords + vocabSize);
            const pWordGivenHam = (countInHam + 1) / (this.totalHamWords + vocabSize);

            spamLogScore += Math.log(pWordGivenSpam);
            hamLogScore += Math.log(pWordGivenHam);

            // Compute spam ratio for this word
            const spamRatio = pWordGivenSpam / pWordGivenHam;
            const logRatio = Math.log(spamRatio);

            wordContributions.push({
                word: token,
                countInSpam,
                countInHam,
                pWordGivenSpam,
                pWordGivenHam,
                logRatio,
                spamRatio
            });
        });

        // Convert log probabilities back to standard probability
        const logDiff = hamLogScore - spamLogScore;
        let probabilitySpam = 1 / (1 + Math.exp(logDiff));
        
        if (isNaN(probabilitySpam)) {
            probabilitySpam = logDiff < 0 ? 1 : 0;
        }

        const label = probabilitySpam >= 0.5 ? 'spam' : 'ham';
        const probability = label === 'spam' ? probabilitySpam : (1 - probabilitySpam);

        // Sort contributions: strongest spam indicators first
        wordContributions.sort((a, b) => b.logRatio - a.logRatio);

        return {
            label,
            probability,
            spamProbability: probabilitySpam,
            rawSpamLogScore: spamLogScore,
            rawHamLogScore: hamLogScore,
            wordContributions
        };
    }

    /**
     * Evaluate model performance on a test dataset
     */
    evaluate(testData) {
        let tp = 0; // True Positive
        let tn = 0; // True Negative
        let fp = 0; // False Positive
        let fn = 0; // False Negative

        testData.forEach(item => {
            const pred = this.predict(item.text);
            const actual = item.label;

            if (actual === 'spam') {
                if (pred.label === 'spam') tp++;
                else fn++;
            } else {
                if (pred.label === 'ham') tn++;
                else fp++;
            }
        });

        const total = testData.length;
        const accuracy = total > 0 ? (tp + tn) / total : 0;
        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

        return {
            accuracy,
            precision,
            recall,
            f1Score: f1,
            confusionMatrix: { tp, tn, fp, fn },
            total
        };
    }
}

// Built-in seed dataset with 40 diverse emails
const SEED_DATASET = [
    // SPAM: Phishing
    {
        label: 'spam',
        category: 'phishing',
        text: 'Dear Customer, your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately to restore access. Click here: http://secure-bank-login-update.com/verify.'
    },
    {
        label: 'spam',
        category: 'phishing',
        text: 'URGENT: Your Netflix account subscription has failed to renew. Update your payment details within 24 hours to prevent account cancellation: http://netflix-payment-verify.support/update.'
    },
    {
        label: 'spam',
        category: 'phishing',
        text: 'Security Alert: We detected an unauthorized login attempt on your account from IP 192.168.1.105 (Location: Russia). If this was not you, reset your password now: http://login-security-recovery.net/reset.'
    },
    {
        label: 'spam',
        category: 'phishing',
        text: 'Important notice regarding your PayPal account. We have noticed unusual activity on a credit card linked to your PayPal account. To secure your account, log in here: http://paypal-resolutioncenter-login.com/webscr.'
    },
    {
        label: 'spam',
        category: 'phishing',
        text: 'Google Security warning! Someone knows your password. Your account password was changed from a computer in Beijing. Reset your Google credentials immediately at http://google-account-recovery-centre.com.'
    },
    
    // SPAM: Lottery & Financial Scams
    {
        label: 'spam',
        category: 'financial',
        text: 'CONGRATULATIONS! Your email address has won $1,500,000.00 in the Google Promo Lottery. Ticket number: 9384-2049. To claim your cash prize, contact agent Mr. Nelson at claim-lottery@gmail.com with your bank details.'
    },
    {
        label: 'spam',
        category: 'financial',
        text: 'Get Rich Quick! Invest only $50 in crypto today and watch it grow to $5,000 in just 48 hours! 100% guaranteed, zero risk. Sign up using our referral link now and receive a free $10 bonus: http://easy-crypto-profits.biz.'
    },
    {
        label: 'spam',
        category: 'financial',
        text: 'Hello, I am Mrs. Mariam Abacha, widow of the late military ruler. I have $25 Million USD that I need to transfer out of the country safely. If you assist me, I will give you 30% of the total funds. Reply with your details.'
    },
    {
        label: 'spam',
        category: 'financial',
        text: 'Pre-approved loan offer! Get up to $50,000 deposited directly into your bank account tomorrow. No credit check required. Bad credit is OK! 1.9% interest rate. Click to apply: http://fast-loans-instant-approval.net.'
    },
    {
        label: 'spam',
        category: 'financial',
        text: 'Earn cash from home! Earn $300 to $800 daily working online. No previous experience needed. All you need is a computer and 2 hours a day. Click here to start making money today: http://earn-cash-easy-jobs.com.'
    },
    
    // SPAM: Marketing & Ads
    {
        label: 'spam',
        category: 'marketing',
        text: 'Lose weight fast! Melt away belly fat in 7 days with our new revolutionary keto pills. No diet, no exercise needed. Order now for a limited-time 70% discount: http://slim-keto-pills-discount.com.'
    },
    {
        label: 'spam',
        category: 'marketing',
        text: 'Buy cheap Viagra, Cialis online! Best prices, high quality pills, overnight shipping. Buy without prescription. Click here to browse our catalog and get 20 free pills: http://cheap-meds-pharmacy.net.'
    },
    {
        label: 'spam',
        category: 'marketing',
        text: 'Hot local singles in your neighborhood are waiting to meet you tonight! 100% free signup. See photos and chat live now. Meet girls nearby: http://local-dating-singles-now.com.'
    },
    {
        label: 'spam',
        category: 'marketing',
        text: 'Special promo! Get the latest smartwatch for 90% off retail price. Only $9.99 for the next 50 customers! Order now before we run out of stock: http://smartwatch-clearance-deal.com.'
    },
    {
        label: 'spam',
        category: 'marketing',
        text: 'Super sale alert! Upgrade your home security system today and get 2 outdoor cameras completely FREE. Low monthly fees, professional monitoring. Get a free quote: http://security-monitoring-systems.org.'
    },
    {
        label: 'spam',
        category: 'marketing',
        text: 'Double your website traffic overnight! Buy 10,000 real visitors for just $29. Boost your search engine rankings and increase sales. Money-back guarantee: http://boost-seo-web-traffic.com.'
    },

    // SPAM: Urgency & Shipping
    {
        label: 'spam',
        category: 'urgency',
        text: 'FedEx Alert: Your package delivery has failed because your address is incomplete. To update your address and reschedule delivery, pay $1.50 custom fee at: http://fedex-package-tracking-update.com.'
    },
    {
        label: 'spam',
        category: 'urgency',
        text: 'Your UPS package is waiting for delivery. Confirm your shipping address and pay the outstanding delivery charge to release the item: http://ups-parcel-delivery-verify.com.'
    },
    {
        label: 'spam',
        category: 'urgency',
        text: 'Immediate action required. Your email storage quota is full. You will not receive new emails until you upgrade. Click here to upgrade your mailbox for free: http://upgrade-email-storage-portal.com.'
    },
    {
        label: 'spam',
        category: 'urgency',
        text: 'Urgent notice from the tax office. You have an outstanding tax refund of $450.70 waiting to be claimed. Fill out the refund form immediately at: http://tax-refund-claim-portal.gov-online.net.'
    },

    // HAM: Business & Work
    {
        label: 'ham',
        category: 'work',
        text: 'Hi everyone, here is the updated project timeline for the Q3 release. We have shifted the database migration to next Tuesday. Please review the tasks in Jira and assign yourself.'
    },
    {
        label: 'ham',
        category: 'work',
        text: 'Can we schedule a quick call today at 3:00 PM to sync on the client feedback? I want to make sure we address the UI issues before the demo tomorrow morning. Let me know if that works.'
    },
    {
        label: 'ham',
        category: 'work',
        text: 'Please find attached the financial report and budget estimates for the next fiscal year. Let me know if you have any questions or see any discrepancies before I present it to the board.'
    },
    {
        label: 'ham',
        category: 'work',
        text: 'Hi Sarah, the code review for your pull request is complete. I left a few minor comments about variable naming and error handling, but overall the implementation looks clean and ready.'
    },
    {
        label: 'ham',
        category: 'work',
        text: 'Dear Team, please submit your expense reports for this month by Friday afternoon. Ensure you attach receipts for all travel expenses exceeding $25 to avoid reimbursement delays.'
    },
    {
        label: 'ham',
        category: 'work',
        text: 'Reminder: The quarterly all-hands meeting is tomorrow at 10:00 AM in the main conference room. We will discuss company milestones, roadmap goals, and introduce the new team members.'
    },
    
    // HAM: Personal
    {
        label: 'ham',
        category: 'personal',
        text: 'Hey! Are we still on for dinner tonight? I was thinking we could try that new Italian restaurant downtown. Let me know what time works best for you, I can make a reservation.'
    },
    {
        label: 'ham',
        category: 'personal',
        text: 'Hi Mom, just wanted to let you know that we arrived safely at the hotel. The flight was a bit delayed, but the weather here is beautiful. I will call you tomorrow morning!'
    },
    {
        label: 'ham',
        category: 'personal',
        text: 'Thanks for lending me your camera last weekend! The photos from the hiking trip turned out amazing. I will drop it off at your place tomorrow evening after work.'
    },
    {
        label: 'ham',
        category: 'personal',
        text: 'Happy Birthday, John! Hope you have a fantastic day filled with fun and relaxation. Let’s get together next week for a beer to celebrate. Talk to you soon!'
    },
    {
        label: 'ham',
        category: 'personal',
        text: 'Hey, did you see the email about the neighborhood block party this Saturday? They are looking for volunteers to bring salads and desserts. Are you planning to go?'
    },

    // HAM: Newsletters & Subscriptions
    {
        label: 'ham',
        category: 'newsletter',
        text: 'Weekly JS News: Chrome 125 launches CSS Anchor Positioning, Node 22 introduces native TypeScript execution, and React 19 enters Release Candidate phase. Read full articles.'
    },
    {
        label: 'ham',
        category: 'newsletter',
        text: 'Github Digest: Here are the top trending open-source repositories this week in Python, JavaScript, and Machine Learning. Check out the latest projects and updates.'
    },
    {
        label: 'ham',
        category: 'newsletter',
        text: 'Medium Daily: You recently read articles about System Design and UI UX Best Practices. Here are some recommendations based on your reading history.'
    },
    {
        label: 'ham',
        category: 'newsletter',
        text: 'Duolingo Daily Streak: Keep your streak alive! It only takes 5 minutes to practice your Spanish today. Complete your lesson and climb the leaderboard.'
    },
    {
        label: 'ham',
        category: 'newsletter',
        text: 'Substack Update: Your subscription to Software Engineering Tidbits has a new post: "How to design resilient microservices under high load." Read online or in app.'
    },

    // HAM: Transactional & Alerts
    {
        label: 'ham',
        category: 'transactional',
        text: 'Your order #7392-0481 has been shipped via UPS. You can track your package details using this link: https://www.ups.com/track/tracking-id-9382049. Estimated delivery is Friday.'
    },
    {
        label: 'ham',
        category: 'transactional',
        text: 'Thank you for your purchase! This is your receipt for order #83049 ($45.99 charged to card ending in 4820). Download your invoice as PDF from your account dashboard.'
    },
    {
        label: 'ham',
        category: 'transactional',
        text: 'Your flight UA-284 from San Francisco (SFO) to New York (JFK) is confirmed. Check-in is now open. Choose your seat and download your mobile boarding pass: https://www.united.com/checkin.'
    },
    {
        label: 'ham',
        category: 'transactional',
        text: 'Your monthly credit card statement is now available online. Total balance due: $340.50, Payment due date: August 5th. Log in to your banking app to schedule payment.'
    },
    {
        label: 'ham',
        category: 'transactional',
        text: 'Appointment Confirmation: You are scheduled for a dental cleaning with Dr. Harris on July 24th at 2:00 PM. Please reply YES to confirm or CALL to reschedule.'
    }
];

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        tokenize,
        NaiveBayesClassifier,
        SEED_DATASET,
        STOP_WORDS
    };
} else {
    window.tokenize = tokenize;
    window.NaiveBayesClassifier = NaiveBayesClassifier;
    window.SEED_DATASET = SEED_DATASET;
    window.STOP_WORDS = STOP_WORDS;
}