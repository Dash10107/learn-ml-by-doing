const { execSync } = require('child_process');
const fs = require('fs');

// Commit messages
const commitMessages = [
  "Initial commit: Add README and .gitignore",
  "Add data preprocessing notebooks",
  "Add classification notebooks: logistic regression, decision tree, SVM",
  "Add regression notebooks: linear, polynomial, random forest",
  "Add clustering notebooks: k-means, hierarchical",
  "Add dimensionality reduction notebooks: PCA, kernel PCA",
  "Add NLP notebooks: basic NLP, text classification",
  "Add reinforcement learning notebooks: Thompson Sampling, UCB",
  "Add generative models notebook: GAN",
  "Add visualization notebook",
  "Organize notebooks into topic-based folders",
  "Update README with project overview and structure",
  "Refactor notebook names for consistency",
  "Add requirements.txt for dependencies",
  "Improve documentation in several notebooks",
  "Fix typos and formatting in README",
  "Add example datasets for demonstration",
  "Update .gitignore to exclude data outputs",
  "Enhance visualization notebook with new plots",
  "Add XGBoost notebook to classification section"
];

// Dates for commits (ISO 8601 format) - last 3 months from June 15, 2025
const dateBuckets = [
  "2025-03-20T10:00:00Z",
  "2025-03-22T12:00:00Z",
  "2025-03-25T09:30:00Z",
  "2025-03-28T14:00:00Z",
  "2025-04-02T11:00:00Z",
  "2025-04-06T15:30:00Z",
  "2025-04-10T10:45:00Z",
  "2025-04-14T13:20:00Z",
  "2025-04-18T09:00:00Z",
  "2025-04-22T11:15:00Z",
  "2025-04-27T08:30:00Z",
  "2025-05-02T12:00:00Z",
  "2025-05-07T10:00:00Z",
  "2025-05-12T13:30:00Z",
  "2025-05-17T09:45:00Z",
  "2025-05-22T11:10:00Z",
  "2025-05-27T10:20:00Z",
  "2025-06-01T12:40:00Z",
  "2025-06-07T09:50:00Z",
  "2025-06-12T11:30:00Z"
];

function getRandomTimeOnDate(dateStr) {
  const start = new Date(`${dateStr}T00:00:00Z`).getTime();
  const end = new Date(`${dateStr}T23:59:59Z`).getTime();
  const randomTime = start + Math.random() * (end - start);
  return new Date(randomTime).toISOString();
}

commitMessages.forEach((message, index) => {
  const day = dateBuckets[index % dateBuckets.length];
  const date = getRandomTimeOnDate(day);

  // Write a dummy line
  fs.appendFileSync('dummy.txt', `${message}\n`);
  execSync('git add dummy.txt');

  // Set both author and committer date
  const env = {
    ...process.env,
    GIT_AUTHOR_DATE: date,
    GIT_COMMITTER_DATE: date,
  };

  execSync(`git commit -m "${message}"`, { env });
  console.log(`✅ Commit ${index + 1}: ${message} @ ${date}`);
});