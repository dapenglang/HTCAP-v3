import argparse, json, torch
from eval_pipeline import PurifierWithHead, get_dataloaders, eval_clean, eval_under_attack
from smoothing import estimate_certificate
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--low-ratio", type=float, default=0.25)
    p.add_argument("--mid-ratio", type=float, default=0.5)
    p.add_argument("--tv-lambda", type=float, default=0.05)
    p.add_argument("--attack", type=str, default="none", choices=["none","pgd"])
    p.add_argument("--norm", type=str, default="linf", choices=["linf","l2"])
    p.add_argument("--eps", type=str, default="8/255")
    p.add_argument("--alpha", type=str, default="2/255")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rs-samples", type=int, default=0)
    p.add_argument("--rs-sigma", type=float, default=0.15)
    return p.parse_args()
def ffloat(x):
    try: return float(eval(x)) if isinstance(x, str) else float(x)
    except Exception: return float(x)
def main():
    args = parse()
    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    model = PurifierWithHead(num_classes=args.num_classes,
                             image_size=args.image_size,
                             low_ratio=args.low_ratio,
                             mid_ratio=args.mid_ratio,
                             tv_lambda=args.tv_lambda).to(device)
    train_loader, test_loader = get_dataloaders(image_size=args.image_size,
                                                num_classes=args.num_classes,
                                                batch_size=args.batch_size)
    clean_acc = eval_clean(model, test_loader, device=device)
    res = {"clean_acc": clean_acc}
    if args.attack != "none":
        acc_adv = eval_under_attack(model, test_loader,
                                    norm=args.norm,
                                    eps=ffloat(args.eps),
                                    alpha=ffloat(args.alpha),
                                    steps=args.steps,
                                    device=device)
        res["adv_acc"] = acc_adv
    if args.rs_samples > 0:
        x, y = next(iter(test_loader))
        x, y = x[:32].to(device), y[:32].to(device)
        pA, pB, radius = estimate_certificate(model, x, y, sigma=args.rs_sigma,
                                              num_samples=args.rs_samples, batch=64, device=device)
        res["rs"] = {"pA": pA, "pB": pB, "radius": radius, "sigma": args.rs_sigma}
    print(json.dumps(res, indent=2))
if __name__ == "__main__":
    main()
