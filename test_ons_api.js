// Test script to check ONS API response structure
// Let's see what the real API returns

const ONS_ENDPOINT = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10/FeatureServer/0/query';

async function testONSAPI() {
    console.log('🧪 Testing ONS API...');
    
    try {
        // Test 1: Get basic info about the service
        console.log('\n1️⃣ Testing service info...');
        const infoParams = new URLSearchParams({
            f: 'json'
        });
        
        const infoResponse = await fetch(`${ONS_ENDPOINT}?${infoParams}`);
        const infoData = await infoResponse.json();
        
        console.log('📊 Service Info:', {
            name: infoData.name,
            type: infoData.type,
            geometryType: infoData.geometryType,
            maxRecordCount: infoData.maxRecordCount,
            fields: infoData.fields?.map(f => ({ name: f.name, type: f.type }))
        });

        // Test 2: Get a single LSOA to see the structure
        console.log('\n2️⃣ Testing single LSOA fetch...');
        const singleParams = new URLSearchParams({
            where: "LSOA21CD like 'E01004%'", // Westminster area
            outSR: '4326',
            f: 'geoJSON',
            outFields: '*',
            resultRecordCount: '1'
        });

        const singleResponse = await fetch(`${ONS_ENDPOINT}?${singleParams}`);
        const singleData = await singleResponse.json();
        
        console.log('🎯 Single LSOA Sample:', {
            type: singleData.type,
            featuresCount: singleData.features?.length,
            sampleFeature: singleData.features?.[0]
        });

        // Test 3: Test Westminster LSOAs specifically
        console.log('\n3️⃣ Testing Westminster LSOAs...');
        const westminsterParams = new URLSearchParams({
            where: "LSOA21NM like 'Westminster%'",
            outSR: '4326',
            f: 'geoJSON',
            outFields: 'LSOA21CD,LSOA21NM,LAT,LONG',
            resultRecordCount: '5' // Just get 5 for testing
        });

        const westminsterResponse = await fetch(`${ONS_ENDPOINT}?${westminsterParams}`);
        const westminsterData = await westminsterResponse.json();
        
        console.log('🏛️ Westminster LSOAs:', {
            count: westminsterData.features?.length,
            features: westminsterData.features?.map(f => ({
                code: f.properties?.LSOA21CD,
                name: f.properties?.LSOA21NM,
                lat: f.properties?.LAT,
                long: f.properties?.LONG,
                geometryType: f.geometry?.type,
                coordinatesLength: f.geometry?.coordinates?.[0]?.length
            }))
        });

        // Test 4: Check available London boroughs
        console.log('\n4️⃣ Testing London borough patterns...');
        const londonBoroughs = [
            'Westminster', 'Camden', 'Islington', 'Hackney', 'Tower Hamlets',
            'Southwark', 'Lambeth', 'City of London'
        ];

        for (const borough of londonBoroughs.slice(0, 3)) { // Test first 3
            const boroughParams = new URLSearchParams({
                where: `LSOA21NM like '${borough}%'`,
                outSR: '4326',
                f: 'json',
                returnCountOnly: 'true'
            });

            const boroughResponse = await fetch(`${ONS_ENDPOINT}?${boroughParams}`);
            const boroughData = await boroughResponse.json();
            
            console.log(`📍 ${borough}: ${boroughData.count || 0} LSOAs`);
        }

        return {
            success: true,
            serviceInfo: infoData,
            sampleData: singleData,
            westminsterData: westminsterData
        };

    } catch (error) {
        console.error('❌ API Test Failed:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

// Test CORS and response
async function testCORS() {
    console.log('\n🌐 Testing CORS and accessibility...');
    
    try {
        const testUrl = `${ONS_ENDPOINT}?f=json&where=1=1&resultRecordCount=1`;
        const response = await fetch(testUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
        
        console.log('📡 Response Status:', response.status);
        console.log('📋 Response Headers:', Object.fromEntries(response.headers.entries()));
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ CORS Test: SUCCESS');
            console.log('📊 Response Type:', typeof data);
            console.log('🔑 Response Keys:', Object.keys(data));
        } else {
            console.log('❌ CORS Test: FAILED');
            console.log('📄 Response Text:', await response.text());
        }
        
    } catch (error) {
        console.log('❌ CORS Error:', error.message);
        
        if (error.message.includes('CORS')) {
            console.log('🚫 CORS is blocking the request');
            console.log('💡 Possible solutions:');
            console.log('   - Use a CORS proxy');
            console.log('   - Make requests from a server');
            console.log('   - Use JSONP if supported');
        }
    }
}

// Alternative test with JSONP
async function testJSONP() {
    console.log('\n🔄 Testing JSONP approach...');
    
    return new Promise((resolve) => {
        const script = document.createElement('script');
        const callbackName = 'onsPCallback' + Date.now();
        
        window[callbackName] = function(data) {
            console.log('✅ JSONP Success:', data);
            document.head.removeChild(script);
            delete window[callbackName];
            resolve(data);
        };
        
        const params = new URLSearchParams({
            where: "LSOA21NM like 'Westminster%'",
            f: 'json',
            callback: callbackName,
            resultRecordCount: '1'
        });
        
        script.src = `${ONS_ENDPOINT}?${params}`;
        script.onerror = () => {
            console.log('❌ JSONP Failed');
            document.head.removeChild(script);
            delete window[callbackName];
            resolve(null);
        };
        
        document.head.appendChild(script);
        
        // Timeout after 10 seconds
        setTimeout(() => {
            if (window[callbackName]) {
                console.log('⏰ JSONP Timeout');
                document.head.removeChild(script);
                delete window[callbackName];
                resolve(null);
            }
        }, 10000);
    });
}

// Run all tests
async function runAllTests() {
    console.log('🚀 Starting ONS API Tests...');
    
    // Test CORS first
    await testCORS();
    
    // Test main API
    const apiResult = await testONSAPI();
    
    // Try JSONP if in browser
    if (typeof document !== 'undefined') {
        await testJSONP();
    }
    
    console.log('\n🏁 Test Complete');
    return apiResult;
}

// Export for use in browser or node
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { testONSAPI, testCORS, runAllTests };
} else if (typeof window !== 'undefined') {
    window.ONSAPITest = { testONSAPI, testCORS, testJSONP, runAllTests };
}

// Auto-run if in browser
if (typeof window !== 'undefined') {
    console.log('🌐 Browser environment detected');
    console.log('💡 Run: ONSAPITest.runAllTests()');
} else {
    console.log('⚡ Node.js environment detected');
    runAllTests().then(result => {
        console.log('Final result:', result);
    });
} 