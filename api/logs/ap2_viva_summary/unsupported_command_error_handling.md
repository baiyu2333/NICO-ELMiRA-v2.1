# Unsupported Command Error Handling

Unsupported or unsafe commands such as `throw the object` must not be executed directly by NICO.

Expected behaviour:
- The system should reject or safely decline the command in natural language.
- No throwing trajectory, high-speed motion, or arbitrary motor command should be generated.
- If the request is ambiguous, the system should ask for a safe alternative such as point, reach, touch, or grasp attempt.
- The event should be logged with the command text, reason for rejection, and `real_robot_motion_commanded=false`.

Safety policy used in the current development workflow:
- Debug tools separate plan-only stages from execution stages.
- Wrist, hand, pre-grasp, and approach execution require safety-gated dashboard controls.
- NJF dry-run and IK trace tools are explicitly no-motion.
- Full grasp is not treated as a single uninspectable command during debugging.

Current evidence status: unsupported-command rejection logs are not available in the provided log set. This should be added as a formal negative test case before final evaluation.
