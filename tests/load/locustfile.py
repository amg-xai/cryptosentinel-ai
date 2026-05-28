"""
CryptoSentinel AI — Load Testing Suite

Three user classes simulating realistic analyst behavior:
  AnalystUser:       read-heavy, GET alerts and graph queries
  InvestigatorUser:  mixed read/write, wallet analysis + acknowledgements
  ScannerUser:       contract scanning, less frequent but expensive

Run headless:
  locust -f tests/load/locustfile.py --headless \
    --users 50 --spawn-rate 5 --run-time 3m \
    --host http://localhost:8000 \
    --csv benchmark-data/load_test_baseline

Run with web UI:
  locust -f tests/load/locustfile.py --host http://localhost:8000
  Open http://localhost:8089
"""
import random
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# Sample Ethereum addresses for realistic testing
SAMPLE_ADDRESSES = [
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
    "0x742d35Cc6634C0532925a3b8D4C9C5B9A4e3F52A",
    "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326",
    "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5",
]

VULNERABLE_CONTRACT = """
pragma solidity ^0.8.0;
contract VulnerableBank {
    mapping(address => uint) public balances;
    function withdraw() public {
        uint amount = balances[msg.sender];
        (bool success,) = msg.sender.call.value(amount)("");
        require(success);
        balances[msg.sender] = 0;
    }
    function kill() public {
        selfdestruct(payable(msg.sender));
    }
}
"""

SAFE_CONTRACT = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/access/Ownable.sol";
contract SafeVault is Ownable {
    uint256 public totalDeposits;
    function deposit() public payable onlyOwner {
        totalDeposits += msg.value;
    }
}
"""


def get_dev_token(client) -> str:
    """Get a development JWT token for authenticated requests."""
    response = client.get("/auth/dev-token?role=analyst",
                         name="/auth/dev-token")
    if response.status_code == 200:
        return response.json().get("access_token", "")
    return ""


class AnalystUser(HttpUser):
    """
    Simulates a SOC analyst monitoring the dashboard.
    Reads alerts, checks graph stats, occasional wallet lookups.
    Wait time: 3-8 seconds between tasks (realistic human interaction).
    """
    wait_time = between(3, 8)
    weight = 3  # 3x more analysts than investigators

    def on_start(self):
        """Called when a user starts — get auth token."""
        self.token = get_dev_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def get_alerts(self):
        """Most common action — check alert feed."""
        self.client.get(
            "/alerts?limit=20",
            headers=self.headers,
            name="/alerts",
        )

    @task(3)
    def get_critical_alerts(self):
        """Check for critical alerts."""
        self.client.get(
            "/alerts/critical",
            headers=self.headers,
            name="/alerts/critical",
        )

    @task(2)
    def check_graph_stats(self):
        """Check overall graph size."""
        self.client.get(
            "/graph/stats/summary",
            headers=self.headers,
            name="/graph/stats/summary",
        )

    @task(1)
    def health_check(self):
        """Occasional health check."""
        self.client.get("/health", name="/health")

    @task(2)
    def lookup_wallet(self):
        """Look up a specific wallet."""
        address = random.choice(SAMPLE_ADDRESSES)
        self.client.get(
            f"/graph/{address}",
            headers=self.headers,
            name="/graph/{address}",
        )


class InvestigatorUser(HttpUser):
    """
    Simulates a threat investigator doing active analysis.
    Analyzes wallets, scans transactions, acknowledges alerts.
    Wait time: 5-15 seconds (more deliberate actions).
    """
    wait_time = between(5, 15)
    weight = 1

    def on_start(self):
        self.token = get_dev_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        # Use investigator role for acknowledge access
        response = self.client.post(
            "/auth/token",
            json={
                "email": "investigator@cryptosentinel.ai",
                "password": "invest123",
            },
            name="/auth/token",
        )
        if response.status_code == 200:
            self.token = response.json().get("access_token", "")
            self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(4)
    def analyze_wallet(self):
        """Analyze a wallet for threat indicators."""
        address = random.choice(SAMPLE_ADDRESSES)
        self.client.post(
            "/analyze/wallet",
            json={"address": address, "include_graph": True},
            headers=self.headers,
            name="/analyze/wallet",
        )

    @task(3)
    def analyze_transaction(self):
        """Score a transaction."""
        self.client.post(
            "/analyze/transaction",
            json={
                "tx_hash": f"0x{'a' * 64}",
                "from_addr": random.choice(SAMPLE_ADDRESSES),
                "to_addr": random.choice(SAMPLE_ADDRESSES),
                "value_eth": round(random.uniform(0, 10), 4),
                "gas": random.randint(21000, 500000),
                "gas_price": random.randint(1_000_000_000, 100_000_000_000),
                "is_contract_call": random.choice([True, False]),
                "is_contract_creation": False,
                "input_data": "0x",
                "block_timestamp": 1779000000.0,
                "chain_name": "ethereum-sepolia",
            },
            headers=self.headers,
            name="/analyze/transaction",
        )

    @task(2)
    def get_alerts(self):
        self.client.get("/alerts", headers=self.headers, name="/alerts")

    @task(1)
    def get_graph(self):
        address = random.choice(SAMPLE_ADDRESSES)
        self.client.get(
            f"/graph/{address}",
            headers=self.headers,
            name="/graph/{address}",
        )


class ScannerUser(HttpUser):
    """
    Simulates a security engineer scanning contracts.
    Infrequent but expensive operations.
    Wait time: 15-30 seconds (contract analysis takes time).
    """
    wait_time = between(15, 30)
    weight = 1

    def on_start(self):
        self.token = get_dev_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def scan_vulnerable_contract(self):
        """Scan a known-vulnerable contract."""
        self.client.post(
            "/scan/contract",
            json={"source_code": VULNERABLE_CONTRACT},
            headers=self.headers,
            name="/scan/contract [vulnerable]",
        )

    @task(1)
    def scan_safe_contract(self):
        """Scan a safe contract for baseline."""
        self.client.post(
            "/scan/contract",
            json={"source_code": SAFE_CONTRACT},
            headers=self.headers,
            name="/scan/contract [safe]",
        )

    @task(2)
    def check_metrics(self):
        """Check Prometheus metrics."""
        self.client.get("/metrics", name="/metrics")
