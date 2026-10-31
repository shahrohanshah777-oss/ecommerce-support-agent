"""Labelled sentences for calibrating the real-LLM-mode groundedness check
(guardrails.check_groundedness(semantic=True), threshold config.GROUNDING_MIN_ENTAILMENT).

Each case is one answer sentence judged against one KB document:
  SUPPORTED   - faithful paraphrases, the way an LLM rewords the evidence
  INVENTED    - plausible same-domain claims the document does not make
  CONTRADICTS - same-topic claims that reverse the document (an embedding-
                similarity check accepted these; the NLI check must reject them)
  REAL_CORRECT - sentences a real model actually wrote, all verified correct
"""
from __future__ import annotations

SUPPORTED = [
    ("03_delivery_sla", "Standard delivery to metro cities usually takes between 2 and 4 business days."),
    ("03_delivery_sla", "If you order before 2 PM, express delivery reaches eligible pin codes the next business day."),
    ("03_delivery_sla", "You will get an SMS and an email if your shipment is delayed."),
    ("04_reverse_pickup", "Reverse pickup means Nykaa collects the item you are returning from your doorstep for free."),
    ("04_reverse_pickup", "If your pin code isn't serviceable, you'll need to ship the item yourself and can get up to Rs 150 back in your Nykaa Wallet."),
    ("04_reverse_pickup", "Opened beauty products and innerwear can't be picked up for a return."),
    ("05_warranty_terms", "Hair dryers and trimmers come with a 12-month manufacturer warranty starting from delivery."),
    ("05_warranty_terms", "Physical or liquid damage is not covered by the warranty."),
    ("06_order_cancellation", "You can cancel your order for free while it is still in Placed status."),
    ("06_order_cancellation", "Shipped orders can't be cancelled unless they are flagged as delayed, so you can refuse the delivery instead."),
    ("06_order_cancellation", "Prepaid cancellations are refunded to your original payment method in 5 to 7 business days."),
    ("07_loyalty_points", "You earn 1 reward point for every Rs 100 you spend on delivered orders."),
    ("07_loyalty_points", "Your reward points expire 12 months after they are credited."),
    ("08_payment_failure_retry", "If money was debited but the payment failed, your bank will reverse it automatically within 5 to 7 business days."),
    ("08_payment_failure_retry", "After a failed payment your cart is held for 30 minutes so you can try again."),
    ("08_payment_failure_retry", "Nykaa will never ask you for your CVV or OTP."),
]

INVENTED = [
    ("03_delivery_sla", "Delivery is always free for all orders across India."),
    ("03_delivery_sla", "You can choose a specific delivery time slot at checkout."),
    ("03_delivery_sla", "Your order will definitely arrive by tomorrow evening."),
    ("04_reverse_pickup", "Our courier will call you an hour before the pickup."),
    ("04_reverse_pickup", "You can also drop the item off at any Nykaa store instead."),
    ("05_warranty_terms", "You can buy an extended warranty for electronics at checkout."),
    ("05_warranty_terms", "Nykaa will replace a broken appliance at your home within 48 hours."),
    ("06_order_cancellation", "You will receive a 10% discount coupon for the inconvenience."),
    ("06_order_cancellation", "Cancellation requests are reviewed by our partner banks."),
    ("06_order_cancellation", "We will send you a replacement product free of charge."),
    ("07_loyalty_points", "Reward points can be transferred to a friend's account."),
    ("07_loyalty_points", "Premium members get free gift wrapping on every order."),
    ("08_payment_failure_retry", "You can also pay using cryptocurrency."),
    ("08_payment_failure_retry", "Our team will call you to complete the payment over the phone."),
    ("08_payment_failure_retry", "Nykaa offers a price-match guarantee on all products."),
]

CONTRADICTS = [
    ("06_order_cancellation", "Shipped orders can always be cancelled free of charge."),
    ("08_payment_failure_retry", "Please share your full card number and CVV so we can verify the payment."),
    ("07_loyalty_points", "Reward points never expire."),
]

# Real model output (openai/gpt-oss-20b via Groq, 2026-10-31) for the Task 4
# in-scope calibration queries (demos/task04_calibrate.py IN_SCOPE; none is in an
# eval set). Each sentence was checked by hand against its KB document and is
# CORRECT. Scored against the same evidence the generator saw (retrieval is
# deterministic, so the demo rebuilds it offline from the query).
REAL_CORRECT = [
    ("How many days do I have to return a pair of shoes?",
     "You have 15 days from the delivery date to return a pair of shoes."),
    ("When will I get my money back for a cash on delivery order?",
     "After the returned item passes the quality check at the Nykaa warehouse, you can receive the refund as a "
     "Nykaa Wallet credit within 24 hours, or via bank transfer (NEFT/IMPS) within 5 to 7 business days."),
    ("How long does standard delivery take to Bangalore?",
     "Standard delivery takes 2 to 4 business days to Bangalore."),
    ("Is there a warranty on a hair dryer?",
     "Yes, hair dryers come with a 12-month manufacturer warranty from the delivery date, which can be serviced "
     "at the brand's authorised service centres with your Nykaa invoice as proof of purchase."),
    ("Can I cancel my order after it has been shipped?",
     "No, you cannot cancel an order after it has shipped unless the shipment is flagged as delayed."),
    ("Can I cancel my order after it has been shipped?",
     "In that case you should refuse delivery or initiate a return once it arrives."),
    ("My payment failed but money was deducted",
     "If a payment fails but money has been debited, no order is created and the bank or payment gateway will "
     "automatically reverse the amount within 5 to 7 business days."),
    ("Can I exchange a dress for a bigger size?", "Yes, you can exchange a dress for a bigger size."),
    ("Can I exchange a dress for a bigger size?",
     "The exchange is allowed within 15 days of delivery, provided the dress is unworn and still has its original tags."),
    ("Can I exchange a dress for a bigger size?",
     "The new size must be in stock; otherwise you'll be offered a refund."),
    ("The lipstick I received was broken",
     "You should raise a claim within 48 hours of delivery, providing clear photos of the broken lipstick, "
     "the outer box, and the shipping label."),
    ("The lipstick I received was broken",
     "The claim will be resolved within 3 business days with a free replacement or a full refund."),
    ("Do you ship to Dubai?", "No, Nykaa only delivers to addresses within India."),
    ("How do I escalate my complaint?", "Reply to your ticket email with the word ESCALATE to request escalation."),
]
