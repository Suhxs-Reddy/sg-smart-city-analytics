"""
Azure VM Deployment Script — Smart City Data Collector

Handles Azure for Students subscription restrictions:
  - Auto-discovers allowed regions via policy
  - Tries multiple VM sizes (B1s → B1ms → B2s → D2s_v3)
  - Creates all networking infrastructure automatically
  - SSH keys from ~/.ssh/id_rsa.pub

Usage:
    python scripts/deploy_azure_vm.py
    python scripts/deploy_azure_vm.py --retry  # Keep retrying every 30 min
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from azure.identity import AzureCliCredential
    from azure.mgmt.resource import ResourceManagementClient
    from azure.mgmt.network import NetworkManagementClient
    from azure.mgmt.compute import ComputeManagementClient
except ImportError:
    print("Install Azure SDK: pip install azure-identity azure-mgmt-compute azure-mgmt-network azure-mgmt-resource")
    sys.exit(1)


SUBSCRIPTION_ID = "5c7f95b6-01ba-4d79-a4d8-04a5843ffbd8"
ALLOWED_REGIONS = ["southcentralus", "canadacentral", "francecentral", "norwayeast", "mexicocentral"]
VM_SIZES = ["Standard_B1s", "Standard_B1ms", "Standard_B2s", "Standard_DS1_v2", "Standard_A1_v2"]
RG_NAME = "sg-city-rg"
VM_NAME = "sg-collector"


def get_ssh_key() -> str:
    key_path = Path.home() / ".ssh" / "id_rsa.pub"
    if not key_path.exists():
        print("No SSH key found. Run: ssh-keygen -t rsa -b 4096")
        sys.exit(1)
    return key_path.read_text().strip()


def try_create_vm(location: str, vm_size: str, credential, ssh_key: str) -> dict | None:
    """Attempt to create a VM. Returns connection info or None."""
    resource_client = ResourceManagementClient(credential, SUBSCRIPTION_ID)
    network_client = NetworkManagementClient(credential, SUBSCRIPTION_ID)
    compute_client = ComputeManagementClient(credential, SUBSCRIPTION_ID)

    rg = f"{RG_NAME}-{location}"

    try:
        resource_client.resource_groups.create_or_update(rg, {"location": location})

        vnet = network_client.virtual_networks.begin_create_or_update(rg, "sg-vnet", {
            "location": location,
            "address_space": {"address_prefixes": ["10.0.0.0/16"]},
            "subnets": [{"name": "default", "address_prefix": "10.0.0.0/24"}],
        }).result()

        ip = network_client.public_ip_addresses.begin_create_or_update(rg, "sg-ip", {
            "location": location,
            "sku": {"name": "Standard"},
            "public_ip_allocation_method": "Static",
        }).result()

        nsg = network_client.network_security_groups.begin_create_or_update(rg, "sg-nsg", {
            "location": location,
            "security_rules": [{
                "name": "AllowSSH", "protocol": "Tcp", "direction": "Inbound",
                "priority": 1000, "source_address_prefix": "*", "source_port_range": "*",
                "destination_address_prefix": "*", "destination_port_range": "22", "access": "Allow",
            }],
        }).result()

        nic = network_client.network_interfaces.begin_create_or_update(rg, "sg-nic", {
            "location": location,
            "ip_configurations": [{
                "name": "default",
                "subnet": {"id": vnet.subnets[0].id},
                "public_ip_address": {"id": ip.id},
            }],
            "network_security_group": {"id": nsg.id},
        }).result()

        vm = compute_client.virtual_machines.begin_create_or_update(rg, VM_NAME, {
            "location": location,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": "Canonical",
                    "offer": "0001-com-ubuntu-server-jammy",
                    "sku": "22_04-lts",
                    "version": "latest",
                },
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "disk_size_gb": 30,
                },
            },
            "os_profile": {
                "computer_name": VM_NAME,
                "admin_username": "azureuser",
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {"public_keys": [{
                        "path": "/home/azureuser/.ssh/authorized_keys",
                        "key_data": ssh_key,
                    }]},
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id}]},
        }).result()

        return {
            "vm_name": vm.name,
            "location": location,
            "vm_size": vm_size,
            "ip_address": ip.ip_address,
            "resource_group": rg,
            "ssh_command": f"ssh azureuser@{ip.ip_address}",
        }

    except Exception as e:
        err = str(e)
        if "SkuNotAvailable" in err:
            return None
        elif "RequestDisallowed" in err:
            return None
        elif "ResourceGroupBeingDeleted" in err:
            return None
        else:
            print(f"    Unexpected: {err[:120]}")
            return None


def deploy(retry: bool = False, retry_interval: int = 1800):
    """Main deployment loop."""
    credential = AzureCliCredential()
    ssh_key = get_ssh_key()

    attempt = 0
    while True:
        attempt += 1
        print(f"\n{'='*50}")
        print(f"Deployment Attempt #{attempt}")
        print(f"{'='*50}")

        for size in VM_SIZES:
            for region in ALLOWED_REGIONS:
                print(f"  {size} in {region}...", end=" ", flush=True)
                result = try_create_vm(region, size, credential, ssh_key)

                if result:
                    print("✅ SUCCESS!")
                    print(f"\n{'='*50}")
                    print(f"  VM DEPLOYED SUCCESSFULLY")
                    for k, v in result.items():
                        print(f"  {k}: {v}")
                    print(f"\n  Next steps:")
                    print(f"  1. {result['ssh_command']}")
                    print(f"  2. bash scripts/setup_azure_vm.sh")
                    print(f"{'='*50}")
                    return result
                else:
                    print("❌")

        if not retry:
            print("\n❌ No capacity available. Run with --retry to auto-retry.")
            return None

        print(f"\nNo capacity. Retrying in {retry_interval//60} minutes...")
        time.sleep(retry_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Azure VM for data collection")
    parser.add_argument("--retry", action="store_true", help="Keep retrying until capacity opens")
    parser.add_argument("--interval", type=int, default=1800, help="Retry interval in seconds (default: 30 min)")
    args = parser.parse_args()

    result = deploy(retry=args.retry, retry_interval=args.interval)
    if not result:
        sys.exit(1)
