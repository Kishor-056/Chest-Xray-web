// Quick fix to reset API URL to ngrok
// Run this in browser console: localStorage.removeItem('appSettings'); location.reload();

console.log("Current localStorage settings:");
try {
    const settings = localStorage.getItem('appSettings');
    if (settings) {
        console.log(JSON.parse(settings));
    } else {
        console.log("No settings found");
    }
} catch (e) {
    console.error(e);
}

console.log("\nTo fix the connection issue:");
console.log("1. Open browser console (F12)");
console.log("2. Run: localStorage.removeItem('appSettings')");
console.log("3. Refresh the page");
console.log("\nOr click the 'Reset to Default' button in Settings");
