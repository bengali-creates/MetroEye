/**
 * Backend API Testing Script (Node.js)
 * =====================================
 *
 * Tests the Node.js backend API endpoints to ensure they're working correctly.
 *
 * Usage:
 *     node test-backend.js
 *
 * Requirements:
 *     npm install axios (or uses fetch in Node.js 18+)
 */

const BASE_URL = 'http://localhost:8000';
const API_URL = `${BASE_URL}/api/alerts`;

// Use fetch (Node.js 18+) or require axios if older
const fetch = globalThis.fetch || require('node-fetch');

// Utility functions
function printSection(title) {
  console.log('\n' + '='.repeat(60));
  console.log(`  ${title}`);
  console.log('='.repeat(60));
}

function formatJSON(obj) {
  return JSON.stringify(obj, null, 2);
}

// Test 1: Health Check
async function testHealthCheck() {
  printSection('Test 1: Health Check');

  try {
    const response = await fetch(`${BASE_URL}/health`);
    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Response: ${formatJSON(data)}`);

    if (response.status === 200) {
      console.log('✓ Health check passed!');
      return true;
    } else {
      console.log('✗ Health check failed!');
      return false;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    console.log('Make sure the backend server is running!');
    return false;
  }
}

// Test 2: Create Alert
async function testCreateAlert() {
  printSection('Test 2: Create Alert');

  // Sample alert data (matching Python system format)
  const alertData = {
    track_id: 123,
    camera_id: 'platform_cam_1',
    timestamp: Date.now() / 1000,
    risk_score: 75,
    risk_level: 'high',
    features: {
      speed_mean: 45.3,
      speed_max: 67.8,
      speed_std: 12.1,
      dist_to_edge_min: 35.2,
      dist_to_edge_mean: 45.8,
      torso_angle_mean: 82.5,
      torso_angle_std: 15.3,
      dwell_time_s: 12.5
    },
    llm_analysis: {
      alert_message: 'Person moving quickly near platform edge',
      recommended_action: 'Alert station staff immediately',
      risk_factors: [
        'High speed movement',
        'Close proximity to edge',
        'Erratic torso angle'
      ],
      confidence: 0.87
    }
  };

  try {
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alertData)
    });

    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Response: ${formatJSON(data)}`);

    if (response.status === 201) {
      console.log('✓ Alert created successfully!');
      return data.data._id;
    } else {
      console.log('✗ Failed to create alert!');
      return null;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    return null;
  }
}

// Test 3: Get All Alerts
async function testGetAlerts() {
  printSection('Test 3: Get All Alerts');

  try {
    const response = await fetch(API_URL);
    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Total Alerts: ${data.total || 0}`);
    console.log(`Current Page: ${data.page || 1}`);
    console.log(`Alerts in Response: ${data.count || 0}`);

    if (data.data && data.data.length > 0) {
      console.log(`\nFirst Alert:`);
      console.log(formatJSON(data.data[0]));
    }

    if (response.status === 200) {
      console.log('✓ Successfully fetched alerts!');
      return true;
    } else {
      console.log('✗ Failed to fetch alerts!');
      return false;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    return false;
  }
}

// Test 4: Get Alert by ID
async function testGetAlertById(alertId) {
  printSection('Test 4: Get Specific Alert');

  if (!alertId) {
    console.log('⚠ Skipping - no alert ID provided');
    return false;
  }

  try {
    const response = await fetch(`${API_URL}/${alertId}`);
    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Response: ${formatJSON(data)}`);

    if (response.status === 200) {
      console.log('✓ Successfully fetched alert by ID!');
      return true;
    } else {
      console.log('✗ Failed to fetch alert!');
      return false;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    return false;
  }
}

// Test 5: Update Alert Status
async function testUpdateAlert(alertId) {
  printSection('Test 5: Update Alert Status');

  if (!alertId) {
    console.log('⚠ Skipping - no alert ID provided');
    return false;
  }

  const updateData = {
    status: 'acknowledged',
    acknowledged_by: 'Test User',
    notes: 'Test acknowledgment from Node.js test script'
  };

  try {
    const response = await fetch(`${API_URL}/${alertId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updateData)
    });

    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Response: ${formatJSON(data)}`);

    if (response.status === 200) {
      console.log('✓ Successfully updated alert!');
      return true;
    } else {
      console.log('✗ Failed to update alert!');
      return false;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    return false;
  }
}

// Test 6: Get Statistics
async function testGetStatistics() {
  printSection('Test 6: Get Alert Statistics');

  try {
    const response = await fetch(`${API_URL}/stats/summary`);
    const data = await response.json();

    console.log(`Status Code: ${response.status}`);
    console.log(`Response: ${formatJSON(data)}`);

    if (response.status === 200) {
      console.log('✓ Successfully fetched statistics!');
      return true;
    } else {
      console.log('✗ Failed to fetch statistics!');
      return false;
    }
  } catch (error) {
    console.log(`✗ Error: ${error.message}`);
    return false;
  }
}

// Test 7: Filtered Queries
async function testFilteredQueries() {
  printSection('Test 7: Filtered Queries');

  const filters = [
    ['camera_id=platform_cam_1', 'Filter by camera'],
    ['risk_level=high', 'Filter by risk level'],
    ['status=pending', 'Filter by status'],
    ['limit=5', 'Limit results']
  ];

  let allPassed = true;

  for (const [queryParam, description] of filters) {
    console.log(`\n${description}: ${queryParam}`);
    try {
      const response = await fetch(`${API_URL}?${queryParam}`);
      const data = await response.json();

      console.log(`  Status: ${response.status}`);
      console.log(`  Count: ${data.count || 0}`);

      if (response.status !== 200) {
        allPassed = false;
        console.log('  ✗ Failed');
      } else {
        console.log('  ✓ Passed');
      }
    } catch (error) {
      console.log(`  ✗ Error: ${error.message}`);
      allPassed = false;
    }
  }

  return allPassed;
}

// Test 8: Create Multiple Alerts
async function createMultipleTestAlerts() {
  printSection('Test 8: Create Multiple Test Alerts');

  const cameras = ['platform_cam_1', 'platform_cam_2', 'station_cam_3'];
  const riskLevels = ['low', 'medium', 'high', 'critical'];

  console.log('Creating 5 test alerts...');

  for (let i = 0; i < 5; i++) {
    const alertData = {
      track_id: 100 + i,
      camera_id: cameras[i % cameras.length],
      timestamp: Date.now() / 1000,
      risk_score: 30 + (i * 15),
      risk_level: riskLevels[i % riskLevels.length],
      features: {
        speed_mean: 20 + (i * 5),
        dist_to_edge_min: 50 - (i * 5)
      },
      llm_analysis: {
        alert_message: `Test alert #${i + 1}`,
        recommended_action: `Test action #${i + 1}`
      }
    };

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alertData)
      });

      if (response.status === 201) {
        console.log(`  ✓ Alert ${i + 1}/5 created`);
      } else {
        console.log(`  ✗ Alert ${i + 1}/5 failed`);
      }
    } catch (error) {
      console.log(`  ✗ Alert ${i + 1}/5 error: ${error.message}`);
    }

    // Small delay between requests
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log('✓ Batch creation complete!');
}

// Main test runner
async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('  BACKEND API TEST SUITE (Node.js)');
  console.log('  Testing Node.js Backend for Suspicious Behavior Detection');
  console.log('='.repeat(60));
  console.log(`\nBackend URL: ${BASE_URL}`);
  console.log(`Testing at: ${new Date().toLocaleString()}`);

  const results = [];

  // Run tests sequentially
  const healthPassed = await testHealthCheck();
  results.push(['Health Check', healthPassed]);

  // If health check fails, don't continue
  if (!healthPassed) {
    console.log('\n⚠ Backend server is not responding. Make sure it\'s running:');
    console.log('   cd suspicious-behavior-backend');
    console.log('   npm run dev');
    return;
  }

  const alertId = await testCreateAlert();
  results.push(['Create Alert', alertId !== null]);

  results.push(['Get All Alerts', await testGetAlerts()]);
  results.push(['Get Alert by ID', await testGetAlertById(alertId)]);
  results.push(['Update Alert', await testUpdateAlert(alertId)]);
  results.push(['Get Statistics', await testGetStatistics()]);
  results.push(['Filtered Queries', await testFilteredQueries()]);

  // Create multiple test alerts
  await createMultipleTestAlerts();

  // Summary
  printSection('TEST SUMMARY');

  const passed = results.filter(([, result]) => result).length;
  const total = results.length;

  for (const [testName, result] of results) {
    const status = result ? '✓ PASS' : '✗ FAIL';
    console.log(`${status.padEnd(8)} - ${testName}`);
  }

  console.log(`\nResults: ${passed}/${total} tests passed`);

  if (passed === total) {
    console.log('\n🎉 All tests passed! Backend is working correctly.');
  } else {
    console.log(`\n⚠ ${total - passed} test(s) failed. Check the errors above.`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('Testing complete!');
  console.log('='.repeat(60) + '\n');
}

// Run tests
main().catch(error => {
  console.error('\n✗ Unexpected error:', error);
  process.exit(1);
});
