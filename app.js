/**
 * SpamSentry Main Application Controller
 * Coordinates DOM interactions, model training, live predictions, and metrics visualizations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // State Management
    // ---------------------------------------------------------
    let currentDataset = [...window.SEED_DATASET];
    let classifier = new window.NaiveBayesClassifier(false);
    
    // Pagination states
    let datasetPage = 1;
    const datasetPageSize = 8;
    let filteredDataset = [...currentDataset];

    let vocabPage = 1;
    const vocabPageSize = 40;
    let filteredVocabList = [];
    let selectedVocabWord = null;

    // ---------------------------------------------------------
    // DOM Element Selectors
    // ---------------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const sidebarModelInfo = document.getElementById('sidebar-model-info');
    const toggleStopwords = document.getElementById('toggle-stopwords');

    const emailInput = document.getElementById('email-input');
    const highlightOverlay = document.getElementById('highlight-overlay');
    const presetSelector = document.getElementById('preset-selector');
    const clearBtn = document.getElementById('clear-btn');
    const charCount = document.getElementById('char-count');
    const resultBadgeCard = document.getElementById('result-badge-card');
    const resultBadge = document.getElementById('result-badge');
    const gaugeFill = document.getElementById('gauge-fill');
    const probabilityValue = document.getElementById('probability-value');
    const spamWordsList = document.getElementById('spam-words-list');
    const hamWordsList = document.getElementById('ham-words-list');

    const addSampleForm = document.getElementById('add-sample-form');
    const totalSamplesCount = document.getElementById('total-samples-count');
    const spamSamplesCount = document.getElementById('spam-samples-count');
    const hamSamplesCount = document.getElementById('ham-samples-count');
    const vocabularyCount = document.getElementById('vocabulary-count');
    const datasetSearch = document.getElementById('dataset-search');
    const retrainBtn = document.getElementById('retrain-btn');
    const datasetTableBody = document.getElementById('dataset-table-body');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const paginationInfo = document.getElementById('pagination-info');

    const vocabSearch = document.getElementById('vocab-search');
    const vocabSort = document.getElementById('vocab-sort');
    const vocabFilterClass = document.getElementById('vocab-filter-class');
    const vocabWordsGrid = document.getElementById('vocab-words-grid');
    const vocabPagination = document.getElementById('vocab-pagination');
    const vocabDetailsPanel = document.getElementById('vocab-details-panel');
    const vocabDetailsContent = document.getElementById('vocab-details-content');

    const metricAccuracy = document.getElementById('metric-accuracy');
    const metricPrecision = document.getElementById('metric-precision');
    const metricRecall = document.getElementById('metric-recall');
    const metricF1 = document.getElementById('metric-f1');
    const cmTN = document.getElementById('cm-tn');
    const cmFN = document.getElementById('cm-fn');
    const cmFP = document.getElementById('cm-fp');
    const cmTP = document.getElementById('cm-tp');
    const validationSplit = document.getElementById('validation-split');
    const splitRatioVal = document.getElementById('split-ratio-val');
    const playTrainCount = document.getElementById('play-train-count');
    const playTestCount = document.getElementById('play-test-count');
    const evalRunBtn = document.getElementById('eval-run-btn');

    // Preset Data Store
    const PRESETS = {
        spam_phishing_bank: `Dear Customer, your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately to restore access. Click here: http://secure-bank-login-update.com/verify.`,
        spam_phishing_netflix: `URGENT: Your Netflix account subscription has failed to renew. Update your payment details within 24 hours to prevent account cancellation: http://netflix-payment-verify.support/update.`,
        spam_lottery: `CONGRATULATIONS! Your email address has won $1,500,000.00 in the Google Promo Lottery. Ticket number: 9384-2049. To claim your cash prize, contact agent Mr. Nelson at claim-lottery@gmail.com with your bank details.`,
        spam_crypto: `Get Rich Quick! Invest only $50 in crypto today and watch it grow to $5,000 in just 48 hours! 100% guaranteed, zero risk. Sign up using our referral link now and receive a free $10 bonus: http://easy-crypto-profits.biz.`,
        spam_pills: `Buy cheap Viagra, Cialis online! Best prices, high quality pills, overnight shipping. Buy without prescription. Click here to browse our catalog and get 20 free pills: http://cheap-meds-pharmacy.net.`,
        spam_fedex: `FedEx Alert: Your package delivery has failed because your address is incomplete. To update your address and reschedule delivery, pay $1.50 custom fee at: http://fedex-package-tracking-update.com.`,
        ham_work_timeline: `Hi everyone, here is the updated project timeline for the Q3 release. We have shifted the database migration to next Tuesday. Please review the tasks in Jira and assign yourself.`,
        ham_work_sync: `Can we schedule a quick call today at 3:00 PM to sync on the client feedback? I want to make sure we address the UI issues before the demo tomorrow morning. Let me know if that works.`,
        ham_personal_dinner: `Hey! Are we still on for dinner tonight? I was thinking we could try that new Italian restaurant downtown. Let me know what time works best for you, I can make a reservation.`,
        ham_newsletter_js: `Weekly JS News: Chrome 125 launches CSS Anchor Positioning, Node 22 introduces native TypeScript execution, and React 19 enters Release Candidate phase. Read full articles.`,
        ham_order_shipping: `Your order #7392-0481 has been shipped via UPS. You can track your package details using this link: https://www.ups.com/track/tracking-id-9382049. Estimated delivery is Friday.`
    };

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // ---------------------------------------------------------
    // Core Model Management
    // ---------------------------------------------------------
    function initModel() {
        classifier.train(currentDataset);
        updateDashboardStats();
        runEvaluation();
        triggerAnalysis();
    }

    function updateDashboardStats() {
        const total = currentDataset.length;
        const spam = currentDataset.filter(e => e.label === 'spam').length;
        const ham = total - spam;
        const vocabSize = classifier.vocabulary.size;

        totalSamplesCount.textContent = total;
        spamSamplesCount.textContent = spam;
        hamSamplesCount.textContent = ham;
        vocabularyCount.textContent = vocabSize;
        sidebarModelInfo.textContent = `Trained: ${total} samples`;
    }

    // ---------------------------------------------------------
    // Navigation / Tab Handler
    // ---------------------------------------------------------
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `tab-${tabId}`) {
                    content.classList.add('active');
                }
            });

            updateTitles(tabId);

            if (tabId === 'dataset') {
                renderDatasetTable();
            } else if (tabId === 'vocabulary') {
                setupVocabularyExplorer();
            }
        });
    });

    function updateTitles(tabId) {
        switch(tabId) {
            case 'analyzer':
                pageTitle.textContent = "Email Analyzer";
                pageSubtitle.textContent = "Analyze emails in real-time using Bayesian probability modeling";
                break;
            case 'dataset':
                pageTitle.textContent = "Dataset Manager";
                pageSubtitle.textContent = "Add, remove, and manage spam and ham training samples";
                break;
            case 'vocabulary':
                pageTitle.textContent = "Vocabulary Explorer";
                pageSubtitle.textContent = "Examine learned words and their mathematical spam probabilities";
                break;
            case 'metrics':
                pageTitle.textContent = "Model Performance Analytics";
                pageSubtitle.textContent = "Assess the accuracy, precision, and recall on simulated split test sets";
                break;
        }
    }

    toggleStopwords.addEventListener('change', () => {
        classifier.removeStopwords = toggleStopwords.checked;
        classifier.train(currentDataset);
        updateDashboardStats();
        runEvaluation();
        triggerAnalysis();
    });

    // ---------------------------------------------------------
    // Tab 1: Email Analyzer Logic
    // ---------------------------------------------------------
    emailInput.addEventListener('scroll', () => {
        highlightOverlay.scrollTop = emailInput.scrollTop;
    });

    const debouncedAnalysis = debounce(triggerAnalysis, 150);
    emailInput.addEventListener('input', () => {
        debouncedAnalysis();
    });

    presetSelector.addEventListener('change', () => {
        const selected = presetSelector.value;
        if (PRESETS[selected]) {
            emailInput.value = PRESETS[selected];
            triggerAnalysis();
            presetSelector.value = "";
            emailInput.focus();
        }
    });

    clearBtn.addEventListener('click', () => {
        emailInput.value = "";
        triggerAnalysis();
        emailInput.focus();
    });

    function triggerAnalysis() {
        const text = emailInput.value.trim();
        const charLen = emailInput.value.length;
        const wordLen = text ? text.split(/\s+/).length : 0;
        charCount.textContent = `${charLen} characters | ${wordLen} words`;

        highlightOverlay.innerHTML = generateHighlightsHTML(emailInput.value);
        highlightOverlay.scrollTop = emailInput.scrollTop;

        if (!text) {
            resultBadgeCard.className = "card result-card";
            resultBadge.innerHTML = `<i class="fa-solid fa-circle-question"></i> <span>Awaiting Input</span>`;
            gaugeFill.style.strokeDashoffset = 534;
            gaugeFill.className = "gauge-fill";
            probabilityValue.textContent = "0.0%";
            spamWordsList.innerHTML = `<li class="empty-list-msg">No words detected</li>`;
            hamWordsList.innerHTML = `<li class="empty-list-msg">No words detected</li>`;
            return;
        }

        const prediction = classifier.predict(text);
        
        if (prediction.label === 'spam') {
            resultBadgeCard.className = "card result-card spam";
            resultBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>Spam Detected</span>`;
        } else {
            resultBadgeCard.className = "card result-card ham";
            resultBadge.innerHTML = `<i class="fa-solid fa-check-double"></i> <span>Safe Email (Ham)</span>`;
        }

        const percent = prediction.spamProbability * 100;
        probabilityValue.textContent = `${percent.toFixed(1)}%`;
        const offset = 534 - (534 * prediction.spamProbability);
        gaugeFill.style.strokeDashoffset = offset;

        if (prediction.spamProbability < 0.35) {
            gaugeFill.className = "gauge-fill ham";
        } else if (prediction.spamProbability < 0.75) {
            gaugeFill.className = "gauge-fill warning";
        } else {
            gaugeFill.className = "gauge-fill spam";
        }

        const contributions = prediction.wordContributions || [];
        const uniqueContributions = [];
        const seenWords = new Set();
        contributions.forEach(c => {
            if (!seenWords.has(c.word)) {
                seenWords.add(c.word);
                uniqueContributions.push(c);
            }
        });

        const spamLeaning = uniqueContributions.filter(c => c.logRatio > 0.05).slice(0, 5);
        if (spamLeaning.length > 0) {
            spamWordsList.innerHTML = spamLeaning.map(c => {
                const scoreStr = c.logRatio > 5 ? '>99x' : `${Math.exp(c.logRatio).toFixed(1)}x`;
                return `<li><span class="word">${c.word}</span><span class="val text-spam">${scoreStr}</span></li>`;
            }).join('');
        } else {
            spamWordsList.innerHTML = `<li class="empty-list-msg">No spam-leaning words</li>`;
        }

        const hamLeaning = uniqueContributions.filter(c => c.logRatio < -0.05).sort((a,b) => a.logRatio - b.logRatio).slice(0, 5);
        if (hamLeaning.length > 0) {
            hamWordsList.innerHTML = hamLeaning.map(c => {
                const scoreStr = c.logRatio < -5 ? '>99x' : `${Math.exp(-c.logRatio).toFixed(1)}x`;
                return `<li><span class="word">${c.word}</span><span class="val text-ham">${scoreStr}</span></li>`;
            }).join('');
        } else {
            hamWordsList.innerHTML = `<li class="empty-list-msg">No ham-leaning words</li>`;
        }
    }

    function generateHighlightsHTML(text) {
        if (!text) return '';
        let escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const regex = /([a-zA-Z0-9$%!]+)/g;
        const parts = escaped.split(regex);
        
        const markup = parts.map(part => {
            if (regex.test(part)) {
                const wordClean = part.toLowerCase();
                
                if (classifier.isTrained) {
                    const countInSpam = classifier.spamWordCounts[wordClean] || 0;
                    const countInHam = classifier.hamWordCounts[wordClean] || 0;

                    if (countInSpam > 0 || countInHam > 0) {
                        const vocabSize = classifier.vocabulary.size;
                        const pWordGivenSpam = (countInSpam + 1) / (classifier.totalSpamWords + vocabSize);
                        const pWordGivenHam = (countInHam + 1) / (classifier.totalHamWords + vocabSize);
                        const logRatio = Math.log(pWordGivenSpam / pWordGivenHam);

                        if (logRatio > 0.6) {
                            return `<mark class="spam-high">${part}</mark>`;
                        } else if (logRatio > 0.05) {
                            return `<mark class="spam-med">${part}</mark>`;
                        } else if (logRatio < -0.2) {
                            return `<mark class="ham-high">${part}</mark>`;
                        }
                    }
                }
            }
            return part;
        });
        return markup.join('') + '\n';
    }

    // ---------------------------------------------------------
    // Tab 2: Dataset Manager Logic
    // ---------------------------------------------------------
    addSampleForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const label = document.getElementById('sample-label').value;
        const category = document.getElementById('sample-category').value;
        const text = document.getElementById('sample-text').value.trim();
        
        if (!text) return;

        currentDataset.unshift({ label, category, text });
        document.getElementById('sample-text').value = "";
        
        classifier.train(currentDataset);
        updateDashboardStats();
        runEvaluation();
        
        filteredDataset = [...currentDataset];
        datasetPage = 1;
        renderDatasetTable();
        triggerAnalysis();
    });

    datasetSearch.addEventListener('input', () => {
        const query = datasetSearch.value.trim().toLowerCase();
        if (query) {
            filteredDataset = currentDataset.filter(item => {
                return item.text.toLowerCase().includes(query) || 
                       item.category.toLowerCase().includes(query) || 
                       item.label.toLowerCase().includes(query);
            });
        } else {
            filteredDataset = [...currentDataset];
        }
        datasetPage = 1;
        renderDatasetTable();
    });

    prevPageBtn.addEventListener('click', () => {
        if (datasetPage > 1) {
            datasetPage--;
            renderDatasetTable();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const maxPage = Math.ceil(filteredDataset.length / datasetPageSize) || 1;
        if (datasetPage < maxPage) {
            datasetPage++;
            renderDatasetTable();
        }
    });

    retrainBtn.addEventListener('click', () => {
        retrainBtn.disabled = true;
        const icon = retrainBtn.querySelector('i');
        icon.classList.add('fa-spin');
        
        setTimeout(() => {
            classifier.train(currentDataset);
            updateDashboardStats();
            runEvaluation();
            triggerAnalysis();
            
            icon.classList.remove('fa-spin');
            retrainBtn.disabled = false;
        }, 500);
    });

    function renderDatasetTable() {
        const maxPage = Math.ceil(filteredDataset.length / datasetPageSize) || 1;
        if (datasetPage > maxPage) datasetPage = maxPage;

        const startIndex = (datasetPage - 1) * datasetPageSize;
        const endIndex = Math.min(startIndex + datasetPageSize, filteredDataset.length);
        const pageItems = filteredDataset.slice(startIndex, endIndex);

        paginationInfo.textContent = `Page ${datasetPage} of ${maxPage} (${filteredDataset.length} records)`;

        if (pageItems.length === 0) {
            datasetTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); font-style: italic;">No records found.</td></tr>`;
            return;
        }

        datasetTableBody.innerHTML = pageItems.map((item, index) => {
            const recordNum = startIndex + index + 1;
            const previewText = item.text.length > 75 ? `${item.text.slice(0, 75)}...` : item.text;
            const masterIndex = currentDataset.findIndex(e => e === item);

            return `
                <tr>
                    <td>${recordNum}</td>
                    <td><span class="cell-badge ${item.label}">${item.label}</span></td>
                    <td><span class="cell-category">${item.category}</span></td>
                    <td><span class="cell-text-preview" title="${item.text.replace(/"/g, '&quot;')}">${previewText}</span></td>
                    <td>
                        <button class="btn danger-btn btn-sm delete-record-btn" data-index="${masterIndex}">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        document.querySelectorAll('.delete-record-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const indexToDelete = parseInt(btn.getAttribute('data-index'), 10);
                if (confirm('Are you sure you want to delete this email?')) {
                    currentDataset.splice(indexToDelete, 1);
                    
                    classifier.train(currentDataset);
                    updateDashboardStats();
                    runEvaluation();
                    triggerAnalysis();

                    const query = datasetSearch.value.trim().toLowerCase();
                    if (query) {
                        filteredDataset = currentDataset.filter(item => {
                            return item.text.toLowerCase().includes(query) || 
                                   item.category.toLowerCase().includes(query) || 
                                   item.label.toLowerCase().includes(query);
                        });
                    } else {
                        filteredDataset = [...currentDataset];
                    }
                    renderDatasetTable();
                }
            });
        });
    }

    // ---------------------------------------------------------
    // Tab 3: Vocabulary Explorer Logic
    // ---------------------------------------------------------
    vocabSearch.addEventListener('input', () => { vocabPage = 1; filterAndSortVocabulary(); });
    vocabSort.addEventListener('change', () => { vocabPage = 1; filterAndSortVocabulary(); });
    vocabFilterClass.addEventListener('change', () => { vocabPage = 1; filterAndSortVocabulary(); });

    function setupVocabularyExplorer() {
        filterAndSortVocabulary();
        if (selectedVocabWord) {
            renderWordDetails(selectedVocabWord);
        } else {
            resetDetailsPanel();
        }
    }

    function filterAndSortVocabulary() {
        const query = vocabSearch.value.trim().toLowerCase();
        const sortBy = vocabSort.value;
        const classFilter = vocabFilterClass.value;

        if (!classifier.isTrained) {
            vocabWordsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted);">Model is not trained.</div>`;
            return;
        }

        const vocabWords = Array.from(classifier.vocabulary);
        const vocabSize = classifier.vocabulary.size;

        let wordList = vocabWords.map(word => {
            const countInSpam = classifier.spamWordCounts[word] || 0;
            const countInHam = classifier.hamWordCounts[word] || 0;
            const pWordGivenSpam = (countInSpam + 1) / (classifier.totalSpamWords + vocabSize);
            const pWordGivenHam = (countInHam + 1) / (classifier.totalHamWords + vocabSize);
            const spamRatio = pWordGivenSpam / pWordGivenHam;
            const logRatio = Math.log(spamRatio);

            return {
                word, countInSpam, countInHam, totalCount: countInSpam + countInHam,
                pWordGivenSpam, pWordGivenHam, spamRatio, logRatio
            };
        });

        if (query) {
            wordList = wordList.filter(item => item.word.includes(query));
        }

        if (classFilter === 'spamLeaning') {
            wordList = wordList.filter(item => item.logRatio > 0.05);
        } else if (classFilter === 'hamLeaning') {
            wordList = wordList.filter(item => item.logRatio < -0.05);
        }

        if (sortBy === 'spamminess') {
            wordList.sort((a, b) => b.logRatio - a.logRatio);
        } else if (sortBy === 'hamminess') {
            wordList.sort((a, b) => a.logRatio - b.logRatio);
        } else if (sortBy === 'spamCount') {
            wordList.sort((a, b) => b.countInSpam - a.countInSpam);
        } else if (sortBy === 'hamCount') {
            wordList.sort((a, b) => b.countInHam - a.countInHam);
        } else if (sortBy === 'alphabetical') {
            wordList.sort((a, b) => a.word.localeCompare(b.word));
        }

        filteredVocabList = wordList;
        renderVocabGrid();
    }

    function renderVocabGrid() {
        const maxPage = Math.ceil(filteredVocabList.length / vocabPageSize) || 1;
        if (vocabPage > maxPage) vocabPage = maxPage;

        const startIndex = (vocabPage - 1) * vocabPageSize;
        const endIndex = Math.min(startIndex + vocabPageSize, filteredVocabList.length);
        const pageItems = filteredVocabList.slice(startIndex, endIndex);

        if (pageItems.length === 0) {
            vocabWordsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); font-style: italic; padding: 40px 0;">No matching words in vocabulary.</div>`;
            vocabPagination.innerHTML = '';
            return;
        }

        vocabWordsGrid.innerHTML = pageItems.map(item => {
            let classLean = "";
            if (item.logRatio > 0.6) classLean = "spam-strong";
            else if (item.logRatio < -0.6) classLean = "ham-strong";
            const isSelected = selectedVocabWord && selectedVocabWord.word === item.word ? "selected" : "";

            return `
                <div class="vocab-badge ${classLean} ${isSelected}" data-word="${item.word}">
                    ${item.word}
                </div>
            `;
        }).join('');

        document.querySelectorAll('.vocab-badge').forEach(badge => {
            badge.addEventListener('click', () => {
                const word = badge.getAttribute('data-word');
                const selectedItem = filteredVocabList.find(i => i.word === word);
                if (selectedItem) {
                    selectedVocabWord = selectedItem;
                    document.querySelectorAll('.vocab-badge').forEach(b => b.classList.remove('selected'));
                    badge.classList.add('selected');
                    renderWordDetails(selectedItem);
                }
            });
        });

        renderVocabPaginationMarkup(maxPage);
    }

    function renderVocabPaginationMarkup(maxPage) {
        if (maxPage <= 1) {
            vocabPagination.innerHTML = '';
            return;
        }

        vocabPagination.innerHTML = `
            <button id="vocab-prev" class="btn secondary-btn btn-sm" ${vocabPage === 1 ? 'disabled' : ''}>
                <i class="fa-solid fa-chevron-left"></i> Prev
            </button>
            <span id="vocab-page-info" style="font-size: 12px; color: var(--text-muted)">
                Page ${vocabPage} of ${maxPage} (${filteredVocabList.length} words)
            </span>
            <button id="vocab-next" class="btn secondary-btn btn-sm" ${vocabPage === maxPage ? 'disabled' : ''}>
                Next <i class="fa-solid fa-chevron-right"></i>
            </button>
        `;

        document.getElementById('vocab-prev').addEventListener('click', () => {
            if (vocabPage > 1) {
                vocabPage--;
                renderVocabGrid();
            }
        });

        document.getElementById('vocab-next').addEventListener('click', () => {
            if (vocabPage < maxPage) {
                vocabPage++;
                renderVocabGrid();
            }
        });
    }

    function resetDetailsPanel() {
        vocabDetailsPanel.querySelector('.details-placeholder').classList.remove('hide');
        vocabDetailsContent.classList.add('hide');
    }

    function renderWordDetails(item) {
        vocabDetailsPanel.querySelector('.details-placeholder').classList.add('hide');
        vocabDetailsContent.classList.remove('hide');

        const totalOccurrences = item.countInSpam + item.countInHam;
        const spamPercent = (item.pWordGivenSpam / (item.pWordGivenSpam + item.pWordGivenHam)) * 100;
        const hamPercent = 100 - spamPercent;

        let explanation = "";
        if (item.logRatio > 0.6) {
            explanation = `The word <strong>"${item.word}"</strong> is heavily associated with spam emails. It is found <strong>${Math.exp(item.logRatio).toFixed(1)}x</strong> more frequently in spam than ham messages.`;
        } else if (item.logRatio > 0.05) {
            explanation = `The word <strong>"${item.word}"</strong> leans towards spam. In the current dataset, it shows a mild correlation with spam topics.`;
        } else if (item.logRatio < -0.6) {
            explanation = `The word <strong>"${item.word}"</strong> is a strong indicator of a safe/legitimate email. It appears <strong>${Math.exp(-item.logRatio).toFixed(1)}x</strong> more often in ham than spam.`;
        } else if (item.logRatio < -0.05) {
            explanation = `The word <strong>"${item.word}"</strong> leans ham. It usually suggests safe work or personal email context.`;
        } else {
            explanation = `The word <strong>"${item.word}"</strong> is relatively neutral. Its appearance in an email does not strongly pull the classification in either direction.`;
        }

        vocabDetailsContent.innerHTML = `
            <h3>${item.word}</h3>
            <span class="details-subtitle">Word Statistics</span>

            <div class="details-stats-list">
                <div class="details-stat-item">
                    <span class="details-stat-label">Total Occurrences:</span>
                    <span class="details-stat-val">${totalOccurrences} times</span>
                </div>
                <div class="details-stat-item">
                    <span class="details-stat-label">Frequency in Spam:</span>
                    <span class="details-stat-val text-spam">${item.countInSpam} times</span>
                </div>
                <div class="details-stat-item">
                    <span class="details-stat-label">Frequency in Ham:</span>
                    <span class="details-stat-val text-ham">${item.countInHam} times</span>
                </div>
                <div class="details-stat-item">
                    <span class="details-stat-label">Spam Probability Weight:</span>
                    <span class="details-stat-val">${(item.pWordGivenSpam * 100).toFixed(4)}%</span>
                </div>
                <div class="details-stat-item">
                    <span class="details-stat-label">Ham Probability Weight:</span>
                    <span class="details-stat-val">${(item.pWordGivenHam * 100).toFixed(4)}%</span>
                </div>
            </div>

            <div class="details-chart-title">Probability Distribution</div>
            <div class="distribution-bar-wrapper">
                <div class="dist-bar-spam" style="width: ${spamPercent}%"></div>
                <div class="dist-bar-ham" style="width: ${hamPercent}%"></div>
            </div>
            <div class="distribution-labels">
                <span class="text-spam">Spam: ${spamPercent.toFixed(1)}%</span>
                <span class="text-ham">Ham: ${hamPercent.toFixed(1)}%</span>
            </div>

            <div class="metrics-educational-info" style="margin-top: 10px; background-color: rgba(255, 255, 255, 0.01); border-style: solid;">
                <p style="font-size: 13px; color: var(--text-main); font-weight: normal; line-height: 1.5;">
                    ${explanation}
                </p>
            </div>
        `;
    }

    // ---------------------------------------------------------
    // Tab 4: Performance Analytics Logic
    // ---------------------------------------------------------
    validationSplit.addEventListener('input', () => {
        const split = validationSplit.value;
        splitRatioVal.textContent = `${split}% Train / ${100 - split}% Test`;
        updateSplitCounts();
    });

    evalRunBtn.addEventListener('click', () => {
        evalRunBtn.disabled = true;
        const icon = evalRunBtn.querySelector('i');
        icon.classList.add('fa-fade');
        
        setTimeout(() => {
            runEvaluation();
            icon.classList.remove('fa-fade');
            evalRunBtn.disabled = false;
        }, 400);
    });

    function updateSplitCounts() {
        const total = currentDataset.length;
        const trainRatio = validationSplit.value / 100;
        const trainCount = Math.floor(total * trainRatio);
        const testCount = total - trainCount;

        playTrainCount.textContent = `${trainCount} emails`;
        playTestCount.textContent = `${testCount} emails`;
    }

    function runEvaluation() {
        const total = currentDataset.length;
        if (total < 5) {
            alert("Please input at least 5 training samples to evaluate the model.");
            return;
        }

        const trainRatio = validationSplit.value / 100;
        const trainSize = Math.floor(total * trainRatio);

        const shuffled = [...currentDataset].sort(() => Math.random() - 0.5);
        const trainData = shuffled.slice(0, trainSize);
        const testData = shuffled.slice(trainSize);

        const evalClassifier = new window.NaiveBayesClassifier(toggleStopwords.checked);
        evalClassifier.train(trainData);

        const evaluation = evalClassifier.evaluate(testData);

        metricAccuracy.textContent = `${(evaluation.accuracy * 100).toFixed(1)}%`;
        metricPrecision.textContent = `${(evaluation.precision * 100).toFixed(1)}%`;
        metricRecall.textContent = `${(evaluation.recall * 100).toFixed(1)}%`;
        metricF1.textContent = `${(evaluation.f1Score * 100).toFixed(1)}%`;

        const cm = evaluation.confusionMatrix;
        cmTN.textContent = cm.tn;
        cmFN.textContent = cm.fn;
        cmFP.textContent = cm.fp;
        cmTP.textContent = cm.tp;

        playTrainCount.textContent = `${trainData.length} emails`;
        playTestCount.textContent = `${testData.length} emails`;
    }

    // ---------------------------------------------------------
    // Initial Start Up
    // ---------------------------------------------------------
    initModel();
    updateSplitCounts();
});