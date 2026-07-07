import requests
import os
from langchain_core.tools import tool
import os
MERIDIAN_API = os.getenv("MERIDIAN_API", "http://localhost:5101")

@tool
def get_menu(category: str = "all") -> str:
  
    """Get menu items from Meridian Eats.
    Pass 'all' to get every item on the menu grouped by category.
    Pass a specific category like 'Rice', 'Salad', 'Pasta' to filter.
    Always display item ID, name, and price clearly to the customer.
    Use 'all' when customer asks for full menu, whole menu, or all items."""
    

    try:
        response = requests.get(f"{MERIDIAN_API}/api/food", timeout=5)
        response.raise_for_status()
        foods = response.json()
    except requests.exceptions.RequestException as e:
        return f"Could not reach the menu service: {e}"

    # Filter by category if one was given
    if category.lower() != "all":
        foods = [f for f in foods if f["categoryName"].lower() == category.lower()]

    if not foods:
        return f"No items found in category '{category}'. Try: Rice, Salad, Pasta, Sandwich, Deserts, Cake, Noodles, Rolls, Veg"

    # Build a clean text summary for the LLM to read
    lines = []
    for f in foods:
      lines.append(
     f"ID:{f['id']} | {f['name']} ({f['categoryName']}) - ${f['price']} - Rating: {f['rating']}/5"
    )

    return "\n".join(lines)



@tool
def check_order_status(order_id: str) -> str:
    """Use this when the user asks about their order status, where their food is, or order tracking.
    The order_id is a number like 5 or 12."""

    token = os.getenv("MERIDIAN_TEST_TOKEN")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(
            f"{MERIDIAN_API}/api/orders/{order_id}/track",
            headers=headers,
            timeout=5
        )

        if response.status_code == 404:
            return f"Order {order_id} not found."

        if response.status_code == 401:
            return "Authentication failed — the test account token may have expired."

        response.raise_for_status()
        order = response.json()

    except requests.exceptions.RequestException as e:
        return f"Could not reach the order tracking service: {e}"

    items = ", ".join([f"{item['name']} x{item['quantity']}" for item in order["items"]])

    return (
        f"Order #{order['id']}\n"
        f"Status: {order['status']}\n"
        f"Items: {items}\n"
        f"Total: ${order['totalAmount']}\n"
        f"Delivering to: {order['street']}, {order['city']}, {order['state']}"
    )

@tool
def cancel_order(order_id: str) -> str:
    """Use this when the user wants to cancel an order.
    Only Pending orders can be cancelled.
    The order_id is a number like 3 or 5."""

    token = os.getenv("MERIDIAN_TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.put(
            f"{MERIDIAN_API}/api/orders/{order_id}/cancel",
            headers=headers,
            timeout=5
        )

        if response.status_code == 404:
            return f"Order {order_id} not found."

        if response.status_code == 400:
            # This is the "can't cancel non-pending" message from your .NET API
            return response.json().get("message", "Cannot cancel this order.")

        if response.status_code == 401:
            return "Authentication failed."

        response.raise_for_status()
        return f"Order #{order_id} has been successfully cancelled."

    except requests.exceptions.RequestException as e:
        return f"Could not reach the order service: {e}"


@tool
def get_my_orders(status_filter: str = "all") -> str:
    """Use this when the user asks to see their orders, order history, or recent orders.
    Optionally filter by status: 'Pending', 'Confirmed', 'Preparing', 'Out for delivery', 'Delivered', 'Cancelled', or 'all'."""

    token = os.getenv("MERIDIAN_TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(
            f"{MERIDIAN_API}/api/orders",
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        orders = response.json()

    except requests.exceptions.RequestException as e:
        return f"Could not reach the order service: {e}"

    if not orders:
        return "You have no orders yet."

    # Filter by status if requested
    if status_filter.lower() != "all":
        orders = [o for o in orders if o["status"].lower() == status_filter.lower()]
        if not orders:
            return f"No orders found with status '{status_filter}'."

    lines = []
    for o in orders:
        items = ", ".join([f"{i['foodItemName']} x{i['quantity']}" for i in o["items"]])
        lines.append(
            f"Order #{o['id']} — {o['status']} — ${o['totalAmount']} — {items}"
        )

    return "\n".join(lines)
    
@tool
def add_to_cart(food_item_id: int, quantity: int) -> str:
    """Add a specific food item to the cart.
    ONLY call this after: (1) customer confirmed the item, (2) customer confirmed the quantity,
    (3) customer confirmed the full order summary.
    Never call this without an explicit quantity from the customer."""
    # ... rest of function stays the same
    token = os.getenv("MERIDIAN_TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.post(
            f"{MERIDIAN_API}/api/cart",
            headers=headers,
            json={"foodItemId": food_item_id, "quantity": quantity},
            timeout=5
        )

        if response.status_code == 400:
            return response.json().get("message", "Could not add item to cart.")

        response.raise_for_status()
        return f"Added {quantity} item(s) to cart successfully."

    except requests.exceptions.RequestException as e:
        return f"Could not reach cart service: {e}"


@tool
def place_order(street: str, city: str, state: str, zip_code: str, phone: str) -> str:
    """Place the order using items in cart. 
    ONLY call this after customer has explicitly confirmed all details.
    Never call without a real phone number — do not proceed if phone is missing or invalid."""
    # ... rest of function stays the same

    token = os.getenv("MERIDIAN_TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.post(
            f"{MERIDIAN_API}/api/orders",
            headers=headers,
            json={
                "street": street,
                "city": city,
                "state": state,
                "zipCode": zip_code,
                "phone": phone,
                "grandTotal": 0  # backend calculates from cart
            },
            timeout=5
        )

        if response.status_code == 400:
            return response.json().get("message", "Could not place order.")

        response.raise_for_status()
        result = response.json()
        return f"Order placed successfully! Your order ID is #{result['orderId']}. You can track it anytime by asking me."

    except requests.exceptions.RequestException as e:
        return f"Could not reach order service: {e}"